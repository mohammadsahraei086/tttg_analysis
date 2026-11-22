"""Why is one signal mass harder to separate than its neighbours?  Compare the signal samples.

    python -m diagnostics.signal_compare --cat noEFT --target 800
    python -m diagnostics.signal_compare --cat noEFT --target 800 --masses 500 800 1000 1200 1500

Run from the project root. Read-only (nothing in tttg/ is changed) and no trained model is needed: it only
looks at the input features of the signal samples (ALL events, both halves) and of tt (the dominant
background). The target mass has to lie between two of the masses, because the idea is to ask
"does the target behave like an interpolation of its neighbours?". Prints

  1. overview per mass: selected events, xsec, sum_genweight, weight per event, weighted yield
  2. data quality: NaN / inf / sentinel (<= -90) / constant columns per mass
  3. train half vs test half of every sample (the split is first/second half, not random:
     if the files are ordered the two halves can differ)  -> max KS distance per sample
  4. mass trend: median of every feature vs mass; the features where the target's median is
     furthest from the interpolation of its two neighbours (in units of the target's own spread)
  5. separation power vs tt: single-feature AUC (signal vs tt) per mass; the features whose
     separation is much weaker at the target than at the neighbours
  6. signal vs signal: a small boosted-tree classifier between every pair of masses (AUC matrix).
     If the target sits "between" its neighbours, AUC(lower, upper) > AUC(lower, target), AUC(target, upper).
     Plus the features that tell the target from each neighbour
and saves plots/diagnostics/signal_compare_<cat>_<target>_{distributions,separation}.png
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from tttg.config import CATS, MASSES, PLOT_DIR

DIAG_PLOT_DIR = PLOT_DIR / "diagnostics"
MAX_EVENTS = 20000          # subsample size for KS / AUC / classifier (keeps it fast)
TT = "tt"
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]


# ----------------------------------------------------------------------------- helpers
def _sub(x, n=MAX_EVENTS, seed=0):
    if len(x) <= n:
        return x
    return x[np.random.default_rng(seed).choice(len(x), n, replace=False)]


def _auc(a, b):
    """AUC of 'a (label 1) vs b (label 0)' for one variable; ties are handled by roc_auc_score."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 20 or len(b) < 20:
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])


def _spread(x):
    """robust sigma: IQR / 1.349 (falls back to std)"""
    x = x[np.isfinite(x)]
    q1, q3 = np.percentile(x, [25, 75])
    s = (q3 - q1) / 1.349
    return s if s > 0 else (x.std() if x.std() > 0 else 1.0)


def neighbours(masses, target):
    lo = [m for m in masses if m < target]
    hi = [m for m in masses if m > target]
    if not lo or not hi:
        return None
    return max(lo), min(hi)


def interp(m, m_lo, m_hi, v_lo, v_hi):
    f = (m - m_lo) / (m_hi - m_lo)
    return v_lo + f * (v_hi - v_lo)


def _sep(sig, ref):
    """separation power of one variable, |AUC - 0.5| * 2  (0 = none, 1 = perfect)"""
    a = _auc(sig, ref)
    return np.nan if np.isnan(a) else 2 * abs(a - 0.5)


# ----------------------------------------------------------------------------- sections
def overview(info, out=print):
    out("\n== 1. overview per mass")
    out(f"  {'mass':>6}{'selected':>10}{'xsec':>12}{'sum_genw':>13}{'sel/genw':>11}{'w/event':>11}{'weighted':>12}")
    for m, d in sorted(info.items()):
        out(f"  {m:>6}{d['n']:>10d}{d.get('xsec', np.nan):>12.4g}{d.get('sumgen', np.nan):>13.6g}"
            f"{d.get('n', 0) / d.get('sumgen', np.nan):>11.4g}{d.get('wev', np.nan):>11.4g}{d.get('yield', np.nan):>12.1f}")
    out("  (sel/genw ~ selection efficiency if the generator weights are ~1; a non-monotonic value in mass"
        "\n   means the selected fraction itself behaves oddly for that sample)")


def data_quality(X, names, out=print):
    out("\n== 2. data quality (columns that are not clean in at least one sample)")
    bad = False
    for key, x in X.items():
        nan = np.mean(~np.isfinite(x), axis=0)
        sent = np.mean(x <= -90, axis=0)
        const = np.ptp(np.where(np.isfinite(x), x, 0), axis=0) == 0
        for j in np.where((nan > 0) | (sent > 0) | const)[0]:
            bad = True
            out(f"  {str(key):>8}: {names[j]:<22} nan/inf={nan[j]:.3f}  sentinel={sent[j]:.3f}  constant={bool(const[j])}")
    if not bad:
        out("  all columns are finite, no sentinels, none constant")


def half_check(X, names, out=print, warn=0.05):
    out("\n== 3. first half vs second half of each sample (max KS distance over columns; ~0.02 is noise)")
    res = {}
    for key, x in X.items():
        h = len(x) // 2
        a, b = _sub(x[:h]), _sub(x[h:], seed=1)
        ks = np.array([ks_2samp(a[:, j], b[:, j]).statistic for j in range(x.shape[1])])
        top = np.argsort(-ks)[:3]
        res[key] = ks.max()
        flag = "   <-- halves differ" if ks.max() > warn else ""
        out(f"  {str(key):>8}: max KS {ks.max():.3f}  ({', '.join(f'{names[j]} {ks[j]:.3f}' for j in top)}){flag}")
    return res


def mass_trend(sig, names, target, out=print, top=12):
    masses = sorted(sig)
    nb = neighbours(masses, target)
    out(f"\n== 4. mass trend: median of each feature; ranked by distance of the target (mass {target}) from the"
        f"\n      interpolation of its neighbours, in units of the target's own spread")
    if nb is None:
        out("  target is at the edge of the mass list: no interpolation possible")
        return []
    lo, hi = nb
    med = {m: np.nanmedian(sig[m], axis=0) for m in masses}
    dev = np.array([(med[target][j] - interp(target, lo, hi, med[lo][j], med[hi][j])) / _spread(sig[target][:, j])
                    for j in range(len(names))])
    order = np.argsort(-np.abs(dev))[:top]
    out(f"  neighbours used: {lo} and {hi}")
    out(f"  {'feature':<20}{'dev [sigma]':>12}" + "".join(f"{m:>10}" for m in masses))
    for j in order:
        mark = "".join(f"{med[m][j]:>10.3g}" for m in masses)
        out(f"  {names[j]:<20}{dev[j]:>12.2f}{mark}")
    out("  (|dev| of 0.1 sigma or less is a smooth trend; large |dev| = the target is not 'between' its neighbours"
        "\n   in that feature. Compare with the typical size of the trend between the other masses.)")
    return list(order)


def separation(sig, ref, names, target, out=print, top=12):
    masses = sorted(sig)
    nb = neighbours(masses, target)
    sep = {m: np.array([_sep(_sub(sig[m][:, j]), _sub(ref[:, j], seed=2)) for j in range(len(names))]) for m in masses}
    out(f"\n== 5. single-feature separation power, signal vs {TT}: |AUC-0.5|*2 (0 none, 1 perfect)")
    if nb is None:
        out("  target is at the edge of the mass list: no neighbours to compare with")
        return sep, []
    lo, hi = nb
    drop = 0.5 * (sep[lo] + sep[hi]) - sep[target]
    drop = np.where(np.isfinite(drop), drop, 0)
    order = np.argsort(-drop)[:top]
    out(f"  features where the target separates worse than the mean of {lo} and {hi}:")
    out(f"  {'feature':<20}{'drop':>8}" + "".join(f"{m:>9}" for m in masses))
    for j in order:
        out(f"  {names[j]:<20}{drop[j]:>8.3f}" + "".join(f"{sep[m][j]:>9.3f}" for m in masses))
    best = {m: np.argsort(-np.nan_to_num(sep[m]))[:3] for m in masses}
    out("\n  three most separating features per mass:")
    for m in masses:
        out(f"  {m:>6}: " + ", ".join(f"{names[j]} {sep[m][j]:.2f}" for j in best[m]))
    out(f"\n  mean separation over all columns: " + "  ".join(f"{m}: {np.nanmean(sep[m]):.3f}" for m in masses))
    return sep, list(order)


def pairwise(sig, names, target, out=print, top=6):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    masses = sorted(sig)
    out("\n== 6. signal vs signal: boosted-tree AUC between every pair of masses (50/50 train/test)")
    mat = np.full((len(masses), len(masses)), np.nan)
    for i, a in enumerate(masses):
        for j in range(i + 1, len(masses)):
            b = masses[j]
            xa, xb = _sub(sig[a]), _sub(sig[b], seed=1)
            n = min(len(xa), len(xb))
            X = np.vstack([xa[:n], xb[:n]])
            y = np.r_[np.ones(n), np.zeros(n)]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
            mat[i, j] = mat[j, i] = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    out("  " + " " * 8 + "".join(f"{m:>8}" for m in masses))
    for i, m in enumerate(masses):
        out(f"  {m:>8}" + "".join("       -" if i == j else f"{mat[i, j]:>8.3f}" for j in range(len(masses))))
    nb = neighbours(masses, target)
    if nb:
        lo, hi = nb
        i_t, i_l, i_h = masses.index(target), masses.index(lo), masses.index(hi)
        ok = mat[i_l, i_h] > max(mat[i_l, i_t], mat[i_t, i_h])
        out(f"\n  AUC({lo},{hi}) = {mat[i_l, i_h]:.3f}   AUC({lo},{target}) = {mat[i_l, i_t]:.3f}   "
            f"AUC({target},{hi}) = {mat[i_t, i_h]:.3f}")
        out("  -> the target sits between its neighbours (smooth in mass)" if ok else
            "  -> the target does NOT sit between its neighbours: it differs from both more than they differ"
            " from each other")
        for other in (lo, hi):
            s = np.array([_sep(_sub(sig[target][:, j]), _sub(sig[other][:, j], seed=3)) for j in range(len(names))])
            k = np.argsort(-np.nan_to_num(s))[:top]
            out(f"  features that tell {target} from {other}: " + ", ".join(f"{names[j]} {s[j]:.2f}" for j in k))
    return mat


# ----------------------------------------------------------------------------- plots
def plot_distributions(sig, tt, names, target, cols, cat):
    if not cols:
        return
    masses = sorted(sig)
    cols = cols[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, j in zip(axes.ravel(), cols):
        allv = np.concatenate([sig[m][:, j] for m in masses] + [tt[:, j]])
        allv = allv[np.isfinite(allv)]
        lo, hi = np.percentile(allv, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1
        bins = np.linspace(lo, hi, 50)
        for k, m in enumerate(masses):
            ax.hist(sig[m][:, j], bins=bins, histtype="step", density=True, lw=2.5 if m == target else 1.5,
                    ls=LINESTYLES[k % len(LINESTYLES)], label=f"Signal_{m}")
        ax.hist(tt[:, j], bins=bins, histtype="stepfilled", density=True, alpha=0.25, color="gray", label=TT)
        ax.set_title(names[j])
        ax.set_yticks([])
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(f"{cat}: features where Signal_{target} is not between its neighbours (thick line = target)")
    fig.tight_layout()
    DIAG_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    f = DIAG_PLOT_DIR / f"signal_compare_{cat}_{target}_distributions.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    print(f"\n  saved {f}")


def plot_separation(sep, names, target, cols, cat):
    if not cols:
        return
    masses = sorted(sep)
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, j in enumerate(cols[:6]):
        ax.plot(masses, [sep[m][j] for m in masses], marker="o", ls=LINESTYLES[k % len(LINESTYLES)], label=names[j])
    ax.axvline(target, color="gray", ls=":")
    ax.set_xlabel("signal mass")
    ax.set_ylabel(f"separation vs {TT}  (|AUC-0.5|*2)")
    ax.legend(fontsize=8)
    ax.set_title(f"{cat}: features that separate worse at {target}")
    fig.tight_layout()
    f = DIAG_PLOT_DIR / f"signal_compare_{cat}_{target}_separation.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    print(f"  saved {f}")


# ----------------------------------------------------------------------------- driver
def analyse(sig, tt, names, target, info=None, cat="noEFT", out=print, plots=True):
    """sig: {mass: (n, ncol) array}, tt: (n, ncol) array, info: optional {mass: dict} for section 1."""
    if info:
        overview(info, out)
    X = {m: x for m, x in sig.items()}
    X[TT] = tt
    data_quality(X, names, out)
    half_check(X, names, out)
    cols_trend = mass_trend(sig, names, target, out)
    sep, cols_sep = separation(sig, tt, names, target, out)
    pairwise(sig, names, target, out)
    if plots:
        plot_distributions(sig, tt, names, target, cols_trend, cat)
        plot_separation(sep, names, target, cols_sep, cat)


def _stack(output, cat, sample, features):
    """All events of one sample, same column layout as PrepareFeaturesForTraning.concatenate_features."""
    cols = []
    for var in features:
        v = output["features"][cat][sample][var].value
        cols.append(v.astype(np.float32) if isinstance(v[0], np.ndarray) else v[:, None].astype(np.float32))
    return np.concatenate(cols, axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--target", type=int, default=800, choices=MASSES)
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=[500, 800, 1000, 1200, 1500])
    a = p.parse_args()
    masses = sorted(set(a.masses) | {a.target})

    from diagnostics.score_spikes import column_names
    from tttg.data.features import PrepareFeaturesForTraning
    from tttg.data.weights import WeightManager

    ds = PrepareFeaturesForTraning()
    out_ = ds.output
    feats = ds.training_features
    names = column_names(out_, a.cat, f"Signal_{masses[0]}", feats)
    sig = {m: _stack(out_, a.cat, f"Signal_{m}", feats) for m in masses}
    tt = _stack(out_, "noEFT", TT, feats)       # backgrounds are always taken from noEFT (as in the training)

    wm = WeightManager()
    info = {}
    for m in masses:
        s = f"Signal_{m}"
        d = dict(n=len(sig[m]))
        try:
            d["xsec"] = float(np.asarray(wm.get_weights(s, "xsec")).ravel()[0])
            d["sumgen"] = float(np.asarray(wm.get_weights(s, "sum_genweight")).ravel()[0])
            w = np.asarray(wm.get_weights(s, "xsec", "luminosity", "sum_genweight", "variation",
                                          cat=a.cat, variation="Nominal"), dtype=float)
            d["wev"], d["yield"] = float(w.mean()), float(w.sum())
        except Exception as e:                  # overview is a nicety, never fatal
            print(f"  (could not read weights for {s}: {e})")
        info[m] = d
    print(f"\n{'=' * 78}\n{a.cat}: signal comparison, target Signal_{a.target}, masses {masses}\n{'=' * 78}")
    analyse(sig, tt, names, a.target, info, a.cat)


if __name__ == "__main__":
    main()
