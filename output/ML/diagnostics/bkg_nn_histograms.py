"""NN-output histogram of EVERY background sample: how does each sample occupy the NN-output bins?

Run from the diagnostics folder ...

    cd diagnostics
    python bkg_nn_histograms.py --cat noEFT --masses 500 800

... or from the project root (as the other checks):

    python -m diagnostics.bkg_nn_histograms --cat noEFT --masses 500 800

It mirrors the histogram step of the limit (tttg/histograms/nn_histograms.py) as closely as possible, so what you
see here is what the limit is fed with, before the rebinning. 2-FOLD CROSS-TRAINING version:

  * EVERY selected event of every sample (PrepareFeaturesForTraning.get_folds_with_sample): part 0 (first half) is
    scored by the fold-0 networks, part 1 (second half) by the fold-1 networks, i.e. every event is scored by
    networks that never saw it in training; same per-fold scaler / features as in the limit
  * score = mean of the sigmoid outputs of the whole seed ensemble of that fold (config.model_paths(cat, m, fold))
  * nominal weight xsec * lumi / N_gen (WeightManager, use="all": no factor 2), config.NN_BINS bins on [0, 1)
  * like hist.axis.Regular, a score >= 1.0 (float32 sigmoid saturates!) or NaN is NOT in any bin (it falls into the
    overflow and is lost for the limit); this is counted and reported per sample
  * the bins that pyhf finally uses (rebin(): merged from the high-NN end until the bkg MC-stat error <= 30 %) are
    drawn as dashed lines, if pyhf can be imported

Prints, per (cat, mass):
  1. per sample: events, weight per event, yield, share of the background, effective events, mean score,
     share of the yield in the signal-like region (score >= --hi-cut), events lost in the overflow
  2. per NN bin: signal, total background, bkg MC-stat error, which sample dominates the bin and with how many MC events
  3. warnings: low-statistics, high-weight samples that dominate a bin (these produce spikes / unstable limit),
     samples that are in the event set but not in config.BACKGROUNDS (not used by the limit) and the other way round
  4. fold consistency (NEW): the two folds use different networks and their histograms are summed, so the score
     distributions of fold 0 and fold 1 must agree (mean score and KS test per sample)
  5. comparison with the earlier single-split result (NEW, --no-compare switches it off): per NN bin background,
     MC-stat error and raw MC events of results/diagnostics/bkg_nn_hist_<cat>_Signal_<m>.json (written by the old
     version of this script) next to the 2-fold numbers
Saves (read-only for tttg/, only writes to plots/diagnostics and results/diagnostics). The file names carry the suffix
_2fold so that the results of the old single-split version are NOT overwritten (they are needed for item 5):
  plots/diagnostics/<cat>_Signal_<m>_2fold_bkg_yield_per_sample.png   weighted yield per bin, one panel per sample
  plots/diagnostics/<cat>_Signal_<m>_2fold_bkg_occupancy.png          heat maps: share of each sample's yield per bin | raw MC events per bin
  plots/diagnostics/<cat>_Signal_<m>_2fold_bkg_stack.png              stacked backgrounds (by config.BACKGROUNDS group) + signal
  results/diagnostics/bkg_nn_hist_<cat>_Signal_<m>_2fold.json         all numbers (edges, yield, sum w^2, raw counts per sample, fold check)
"""
import argparse
import json
import sys
from pathlib import Path

# make `tttg` importable when this file is run from inside diagnostics/ (or from anywhere)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from tttg.config import (BACKGROUNDS, CATS, LUMINOSITY, MASSES, NN_BINS, PLOT_DIR, RESULTS_DIR, SIG_SYSTS,
                         model_paths)

DIAG_PLOT_DIR = PLOT_DIR / "diagnostics"
DIAG_RESULTS_DIR = RESULTS_DIR / "diagnostics"
CHUNK = 65536


# ----------------------------------------------------------------------------- loading (needs torch + coffea)
def ensemble_scores(cat, mass, folded):
    """NN score of EVERY selected event, as in nn_histograms.get_histograms (2-fold cross-training).

    folded: {sample: [X_part0, X_part1]} from PrepareFeaturesForTraning.get_folds_with_sample. Part f is scored by
    the ensemble of fold f (mean sigmoid output of its seed members); the two parts are put back in file order.
    Returns (scores {sample: array in file order}, fold_scores {sample: [part0 scores, part1 scores]},
    model files {fold: [paths]})."""
    import torch
    from tttg.training.model import OptimizedHEPClassifier

    ncol = next(iter(folded.values()))[0].shape[1]
    ensembles, paths = [], {}
    for fold in (0, 1):
        files = model_paths(cat, mass, fold=fold)
        if not files:
            raise FileNotFoundError(f"no trained model for {cat} Signal_{mass} fold {fold} "
                                    f"(run tttg.training.train_eval, both folds)")
        members = []
        for p in files:
            m = OptimizedHEPClassifier(input_dim=ncol)
            m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
            members.append(m.eval())
        ensembles.append(members)
        paths[fold] = files

    def score(x, members):
        x = np.asarray(x, dtype=np.float32)
        out = []
        for i in range(0, len(x), CHUNK):
            t = torch.from_numpy(x[i:i + CHUNK])
            out.append(torch.stack([torch.sigmoid(m(t)).flatten() for m in members]).mean(dim=0).numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    scores, fold_scores = {}, {}
    with torch.no_grad():
        for s, parts in folded.items():
            fold_scores[s] = [score(parts[f], ensembles[f]) for f in (0, 1)]    # part f <- networks of fold f
            scores[s] = np.concatenate(fold_scores[s])                          # part 0, then part 1 = file order
    return scores, fold_scores, paths


def nominal_weights(wm, cat, scores):
    """Per-event nominal weight of every event (use="all": xsec * lumi / N_gen, no factor 2), as used to fill the limit
    histograms in nn_histograms.get_histograms. A length mismatch between weights and scores raises an error."""
    out = {}
    for s, sc in scores.items():
        if s.startswith("Signal_"):
            w = wm.get_weights(s, "xsec", "luminosity", "sum_genweight", "variation", cat=cat, variation="Nominal",
                               use="all")
        else:
            w = wm.get_weights(s, "xsec", "luminosity", "sum_genweight", use="all")
        w = np.broadcast_to(np.asarray(w, dtype=float), (len(sc),)).copy()
        out[s] = w
    return out


# ----------------------------------------------------------------------------- analysis (pure numpy)
def group_of(sample):
    for g, lst in BACKGROUNDS.items():
        if sample in lst:
            return g
    return None


def fill(score, w, edges, hi_cut):
    """Weighted histogram with the semantics of hist.axis.Regular: bins [lo, hi), score >= edges[-1], < edges[0] and
    NaN are NOT counted in any bin (overflow / underflow). Also returns raw counts and sum of w^2."""
    score = np.asarray(score, dtype=float)
    w = np.asarray(w, dtype=float)
    nan = np.isnan(score)
    inside = ~nan & (score >= edges[0]) & (score < edges[-1])
    idx = np.digitize(score[inside], edges) - 1
    nb = len(edges) - 1
    yld = np.bincount(idx, weights=w[inside], minlength=nb)
    sw2 = np.bincount(idx, weights=w[inside] ** 2, minlength=nb)
    raw = np.bincount(idx, minlength=nb)
    sumw, sumw2 = w.sum(), (w ** 2).sum()
    good = ~nan
    return {
        "yield": yld, "sumw2": sw2, "nraw": raw,
        "n_events": int(len(score)),
        "n_over": int((~nan & (score >= edges[-1])).sum()),
        "n_under": int((~nan & (score < edges[0])).sum()),
        "n_nan": int(nan.sum()),
        "w_lost": float(w[~inside].sum()),
        "w_mean": float(w.mean()) if len(w) else 0.0,
        "w_max": float(w.max()) if len(w) else 0.0,
        "yield_all": float(sumw),
        "neff": float(sumw ** 2 / sumw2) if sumw2 > 0 else 0.0,
        "mean_score": float(score[good].mean()) if good.any() else float("nan"),
        "frac_hi": float(w[good & (score >= hi_cut)].sum() / sumw) if sumw != 0 else float("nan"),
    }


def merged_edges(edges, sig, bkg_tot, bkg_var):
    """Bin edges after the rebin() of the limit (tttg/limits/pyhf_limits.py); None if pyhf is not importable.
    The merging only depends on the background, so a zero 'syst' dictionary is passed."""
    try:
        from tttg.limits.pyhf_limits import bin_merger
    except Exception as e:  # pyhf missing etc.
        print(f"  (pyhf bin merging not drawn: {type(e).__name__}: {e})")
        return None
    zeros = {f"{n}_unc": np.zeros(len(bkg_tot)) for n in SIG_SYSTS}
    try:
        new_edges, _ = bin_merger(edges, sig, bkg_tot, bkg_var, zeros)
    except Exception as e:
        print(f"  (bin merging failed: {type(e).__name__}: {e})")
        return None
    return [float(x) for x in new_edges]


def analyse(scores, weights, signal, edges, hi_cut=0.5, draw_merged=True):
    bkgs = [s for s in scores if s != signal]
    res = {"signal": signal, "edges": [float(e) for e in edges], "hi_cut": hi_cut,
           "samples": {s: fill(scores[s], weights[s], edges, hi_cut) for s in bkgs},
           "signal_hist": fill(scores[signal], weights[signal], edges, hi_cut)}
    for s in bkgs:
        res["samples"][s]["group"] = group_of(s)

    used = [s for s in bkgs if res["samples"][s]["group"] is not None]          # what the limit actually sums
    res["in_limit"] = used
    res["not_in_limit"] = [s for s in bkgs if s not in used]
    res["missing_in_test"] = sorted({s for lst in BACKGROUNDS.values() for s in lst} - set(bkgs))
    tot = sum(res["samples"][s]["yield"] for s in used) if used else np.zeros(len(edges) - 1)
    var = sum(res["samples"][s]["sumw2"] for s in used) if used else np.zeros(len(edges) - 1)
    res["bkg_total"], res["bkg_var"] = tot, var
    res["merged_edges"] = (merged_edges(edges, res["signal_hist"]["yield"], tot, var)
                           if draw_merged and used and len(edges) - 1 == NN_BINS else None)
    return res


def bin_table(res, min_raw):
    """Per NN bin: S, B, MC-stat error, dominant sample, plus warnings about low-stat dominant samples."""
    edges, used, S = res["edges"], res["in_limit"], res["signal_hist"]["yield"]
    rows, warns = [], []
    for i in range(len(edges) - 1):
        ys = {s: res["samples"][s]["yield"][i] for s in used}
        B = res["bkg_total"][i]
        top = max(ys, key=ys.get) if used and B > 0 else None
        nraw = sum(int(res["samples"][s]["nraw"][i]) for s in used)
        w2 = res["bkg_var"][i]
        neff = B ** 2 / w2 if w2 > 0 else 0.0
        rows.append(dict(i=i, lo=edges[i], hi=edges[i + 1], S=S[i], B=B, err=(np.sqrt(w2) / B if B > 0 else np.inf),
                         nraw=nraw, neff=neff, top=top, share=(ys[top] / B if top else 0.0),
                         top_nraw=(int(res["samples"][top]["nraw"][i]) if top else 0)))
        for s in used:                       # a sample carrying >= 20% of the bin with very few MC events
            n = int(res["samples"][s]["nraw"][i])
            if B > 0 and 0 < n < min_raw and ys[s] / B >= 0.2:
                warns.append((s, i, ys[s] / B, n, ys[s] / n))
    return rows, warns


def print_report(res, min_raw):
    used, S, sig = res["in_limit"], res["signal_hist"], res["signal"]
    totB = sum(res["samples"][s]["yield"].sum() for s in used) or 1.0
    order = sorted(res["samples"], key=lambda s: -res["samples"][s]["yield_all"])
    hc = res["hi_cut"]

    print(f"\n== 1. per background sample (ALL events, 2-fold ensemble score, L = {LUMINOSITY} fb^-1)")
    print(f"  {'sample':<14}{'group':<9}{'N_evt':>9}{'w/event':>11}{'w_max':>10}{'yield':>11}{'%bkg':>7}"
          f"{'N_eff':>9}{'<score>':>9}{f'%y>={hc:g}':>9}{'over/NaN':>10}{'lost y':>9}")
    for s in order:
        d = res["samples"][s]
        print(f"  {s:<14}{(d['group'] or '-'):<9}{d['n_events']:>9d}{d['w_mean']:>11.3g}{d['w_max']:>10.3g}"
              f"{d['yield_all']:>11.4g}{100 * d['yield_all'] / totB if d['group'] else float('nan'):>7.1f}"
              f"{d['neff']:>9.1f}{d['mean_score']:>9.3f}{100 * d['frac_hi']:>9.1f}"
              f"{d['n_over'] + d['n_nan']:>10d}{d['w_lost']:>9.2g}")
    d = S
    print(f"  {sig:<14}{'signal':<9}{d['n_events']:>9d}{d['w_mean']:>11.3g}{d['w_max']:>10.3g}{d['yield_all']:>11.4g}"
          f"{'':>7}{d['neff']:>9.1f}{d['mean_score']:>9.3f}{100 * d['frac_hi']:>9.1f}"
          f"{d['n_over'] + d['n_nan']:>10d}{d['w_lost']:>9.2g}")

    rows, warns = bin_table(res, min_raw)
    print("\n== 2. per NN bin (backgrounds of config.BACKGROUNDS only; dashed lines in the plots = pyhf bins)")
    print(f"  {'bin':>3} {'range':<13}{'signal':>10}{'bkg':>11}{'MCstat':>8}{'N_raw':>8}{'N_eff':>8}   dominant sample")
    for r in rows:
        print(f"  {r['i']:>3} [{r['lo']:.3f},{r['hi']:.3f}){r['S']:>10.3g}{r['B']:>11.4g}{r['err']:>8.2f}"
              f"{r['nraw']:>8d}{r['neff']:>8.1f}   "
              + (f"{r['top']} {100 * r['share']:.0f}% ({r['top_nraw']} MC events)" if r['top'] else "-"))
    if res["merged_edges"]:
        print("  pyhf bins after rebin():", ", ".join(f"{e:.3f}" for e in res["merged_edges"]))

    print(f"\n== 3. warnings")
    n_warn = 0
    for s, i, share, n, wev in warns:
        n_warn += 1
        print(f"  low-stat sample dominates a bin: {s} in bin {i} = {100 * share:.0f}% of the bkg from only {n} MC "
              f"event(s) (weight {wev:.3g} per event) -> spike / unstable limit")
    for s in order:
        d = res["samples"][s]
        if d["n_over"] or d["n_nan"] or d["n_under"]:
            n_warn += 1
            print(f"  {s}: {d['n_over']} event(s) with score >= 1.0, {d['n_under']} < 0, {d['n_nan']} NaN: not in any "
                  f"bin (yield lost {d['w_lost']:.3g})")
    if S["n_over"] or S["n_nan"]:
        n_warn += 1
        print(f"  {sig}: {S['n_over']} event(s) with score >= 1.0 and {S['n_nan']} NaN are not in the signal histogram "
              f"(signal yield lost {S['w_lost']:.3g} of {S['yield_all']:.3g})")
    if res["not_in_limit"]:
        n_warn += 1
        print(f"  in the event set but NOT in config.BACKGROUNDS (not used by the limit): {', '.join(res['not_in_limit'])}")
    if res["missing_in_test"]:
        n_warn += 1
        print(f"  in config.BACKGROUNDS but NOT in the event set: {', '.join(res['missing_in_test'])}")
    if not n_warn:
        print("  none")


def fold_check(fold_scores, res, n_top=6):
    """Do the two folds agree? Their histograms are summed in the limit although they come from different networks.
    Per sample (signal + the n_top backgrounds with the largest yield): number of events, mean score and
    Kolmogorov-Smirnov distance / p-value between the score distribution of part 0 (fold-0 networks) and part 1
    (fold-1 networks). Both parts are statistically equivalent samples, so a small p-value (< ~0.01) for a sample with
    many events means the two ensembles are not equivalent (e.g. one fold trained badly). Unweighted, on raw events."""
    from scipy.stats import ks_2samp
    order = sorted(res["in_limit"], key=lambda s: -res["samples"][s]["yield_all"])[:n_top]
    out = {}
    for s in [res["signal"]] + order:
        a, b = (np.asarray(x, dtype=float) for x in fold_scores[s])
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        ks = ks_2samp(a, b)
        out[s] = {"n0": int(len(a)), "n1": int(len(b)), "mean0": float(a.mean()), "mean1": float(b.mean()),
                  "ks": float(ks.statistic), "p": float(ks.pvalue)}
    return out


def print_fold_check(fc):
    print("\n== 4. fold consistency (score distribution of part 0 [fold-0 networks] vs part 1 [fold-1 networks])")
    print(f"  {'sample':<14}{'N_part0':>9}{'N_part1':>9}{'<score>0':>10}{'<score>1':>10}{'KS dist':>9}{'KS p':>9}")
    for s, d in fc.items():
        flag = "   <-- folds differ?" if d["p"] < 0.01 and min(d["n0"], d["n1"]) >= 200 else ""
        print(f"  {s:<14}{d['n0']:>9d}{d['n1']:>9d}{d['mean0']:>10.3f}{d['mean1']:>10.3f}{d['ks']:>9.3f}{d['p']:>9.3f}{flag}")


def print_compare_old(res, cat, signal, min_raw):
    """Per NN bin: background, MC-stat error and raw MC events of the OLD single-split result (json written by the old
    version of this script, same bins) next to the 2-fold numbers."""
    old_path = DIAG_RESULTS_DIR / f"bkg_nn_hist_{cat}_{signal}.json"
    print(f"\n== 5. comparison with the single-split result ({old_path.name})")
    if not old_path.exists():
        print("  not found (run the old version of this script first, or use --no-compare)")
        return
    with open(old_path) as f:
        old = json.load(f)
    if len(old["edges"]) != len(res["edges"]) or not np.allclose(old["edges"], res["edges"]):
        print("  the old result has other NN bins -> not comparable (use the same --bins)")
        return
    oB, oV = np.asarray(old["bkg_total"], dtype=float), np.asarray(old["bkg_var"], dtype=float)
    oN = np.sum([old["samples"][s]["nraw"] for s in old["in_limit"]], axis=0)
    nB, nV = res["bkg_total"], res["bkg_var"]
    nN = np.sum([res["samples"][s]["nraw"] for s in res["in_limit"]], axis=0)
    err = lambda B, V: np.where(B > 0, np.sqrt(V) / np.where(B > 0, B, 1.0), np.inf)
    oE, nE = err(oB, oV), err(nB, nV)
    print(f"  {'bin':>3} {'range':<13}{'bkg old':>11}{'bkg new':>11}{'MCstat old':>12}{'MCstat new':>12}"
          f"{'N_raw old':>11}{'N_raw new':>11}")
    for i in range(len(res["edges"]) - 1):
        lo, hi = res["edges"][i], res["edges"][i + 1]
        flag = "  <-- still < %d MC events" % min_raw if 0 < nN[i] < min_raw else ""
        print(f"  {i:>3} [{lo:.3f},{hi:.3f}){oB[i]:>11.4g}{nB[i]:>11.4g}{oE[i]:>12.2f}{nE[i]:>12.2f}"
              f"{int(oN[i]):>11d}{int(nN[i]):>11d}{flag}")
    print(f"  total bkg: old {oB.sum():.5g}  new {nB.sum():.5g}  (expected equal within MC fluctuations: "
          f"same physics, twice the events)")
    old_m, new_m = old.get("merged_edges"), res.get("merged_edges")
    if old_m and new_m:
        print(f"  pyhf bins after rebin(): old {len(old_m) - 1}  ->  new {len(new_m) - 1}")


# ----------------------------------------------------------------------------- plots
def _style_edges(ax, res):
    if res["merged_edges"]:
        for e in res["merged_edges"][1:-1]:
            ax.axvline(e, color="gray", ls="--", lw=0.7, alpha=0.7)


def plot_yield_per_sample(res, path, min_raw, ncols=4):
    edges = np.asarray(res["edges"])
    centers, order = 0.5 * (edges[1:] + edges[:-1]), sorted(res["samples"], key=lambda s: -res["samples"][s]["yield_all"])
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.7 * nrows), squeeze=False)
    for ax, s in zip(axes.flat, order):
        d = res["samples"][s]
        y, err, n = d["yield"], np.sqrt(d["sumw2"]), d["nraw"]
        ax.stairs(y, edges, color="C0", lw=1.4)
        ok = y > 0
        ax.errorbar(centers[ok], y[ok], yerr=err[ok], fmt="none", ecolor="C0", lw=0.9)
        low = ok & (n < min_raw)
        ax.plot(centers[low], y[low], "x", color="red", ms=5, label=f"< {min_raw} MC events")
        _style_edges(ax, res)
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        if ok.any():
            ax.set_ylim(y[ok].min() / 5, y[ok].max() * 8)
        ax.set_title(f"{s} [{d['group'] or 'NOT in limit'}]\nN={d['n_events']}  Y={d['yield_all']:.3g}  N_eff={d['neff']:.0f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        if low.any():
            ax.legend(fontsize=6, loc="lower left")
    for ax in list(axes.flat)[len(order):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("NN output", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"events / bin (L={LUMINOSITY} fb$^{{-1}}$)", fontsize=8)
    fig.suptitle(f"{path.stem.split('_bkg')[0]}: weighted NN-output histogram of each background sample "
                 f"(error bars = sqrt(sum w$^2$); dashed = pyhf bins)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_occupancy(res, path):
    edges = np.asarray(res["edges"])
    order = sorted(res["samples"], key=lambda s: -res["samples"][s]["yield_all"])
    frac = np.array([res["samples"][s]["yield"] / max(res["samples"][s]["yield"].sum(), 1e-300) for s in order])
    raw = np.array([res["samples"][s]["nraw"] for s in order], dtype=float)
    nb = len(edges) - 1
    labels = [f"{s} [{res['samples'][s]['group'] or 'NOT in limit'}]" for s in order]
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.55 * nb + 3.2), 0.38 * len(order) + 2.2))
    panels = ((axes[0], frac, "share of the sample's yield per NN bin", None),
              (axes[1], np.ma.masked_equal(raw, 0), "raw MC events per NN bin (white = 0)", LogNorm()))
    for ax, data, title, norm in panels:
        im = ax.imshow(data, aspect="auto", cmap="viridis", norm=norm, origin="upper")
        ax.set_xticks(range(nb))
        ax.set_xticklabels([f"{0.5 * (edges[i] + edges[i + 1]):.2f}" for i in range(nb)], rotation=90, fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(order), fontsize=8)
        ax.set_xlabel("NN output (bin centre)")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        vmax = np.nanmax(data) if data.size else 1
        for r in range(data.shape[0]):
            for c in range(nb):
                v = data[r, c]
                if np.ma.is_masked(v) or v <= 0 or (norm is None and v < 0.01):
                    continue
                txt = f"{100 * v:.0f}" if norm is None else (f"{v:.0f}" if v < 1e4 else f"{v / 1e3:.0f}k")
                dark = (v / vmax) > (0.5 if norm is None else 0.05)
                ax.text(c, r, txt, ha="center", va="center", fontsize=6, color="black" if dark else "white")
    fig.suptitle(f"{path.stem.split('_bkg')[0]}: how each background sample occupies the NN-output bins "
                 f"(rows sorted by yield; left numbers in %)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stack(res, path):
    edges = np.asarray(res["edges"])
    groups = {}
    for s in res["in_limit"]:
        g = res["samples"][s]["group"]
        groups[g] = groups.get(g, np.zeros(len(edges) - 1)) + res["samples"][s]["yield"]
    names = sorted(groups, key=lambda g: -groups[g].sum())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(edges) - 1)
    for k, g in enumerate(names[::-1]):                     # biggest group at the bottom
        ax.bar(edges[:-1], groups[g], width=np.diff(edges), align="edge", bottom=bottom, color=cmap(k % 10),
               edgecolor="white", lw=0.4, label=g)
        bottom += groups[g]
    ax.stairs(res["signal_hist"]["yield"], edges, color="black", lw=1.8, label=f"{res['signal']} (xsec in coffea metadata)")
    _style_edges(ax, res)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    pos = bottom[bottom > 0]
    if len(pos):
        ax.set_ylim(max(pos.min() / 5, 1e-6), pos.max() * 20)
    ax.set_xlabel("NN output")
    ax.set_ylabel(f"events / bin (L={LUMINOSITY} fb$^{{-1}}$)")
    ax.set_title(f"{path.stem.split('_bkg')[0]}: background composition seen by the limit (dashed = pyhf bins)", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_json(res, path):
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()
           if k not in ("samples", "signal_hist")}
    conv = lambda d: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    out["samples"] = {s: conv(d) for s, d in res["samples"].items()}
    out["signal_hist"] = conv(res["signal_hist"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


# ----------------------------------------------------------------------------- main
def run_one(ds, wm, cat, mass, nbins, hi_cut, min_raw, plots=True, compare=True):
    signal = f"Signal_{mass}"
    folded = ds.get_folds_with_sample(signal=signal, cat=cat)
    scores, fold_scores, paths = ensemble_scores(cat, mass, folded)
    weights = nominal_weights(wm, cat, scores)
    print(f"\n{'=' * 100}\n{cat}  {signal}: 2-fold cross-training")
    for fold, files in paths.items():
        print(f"  fold {fold} (scores part {fold}): ensemble of {len(files)} model(s): " + ", ".join(p.name for p in files))
    res = analyse(scores, weights, signal, np.linspace(0.0, 1.0, nbins + 1), hi_cut=hi_cut)
    res["fold_check"] = fold_check(fold_scores, res)
    print_report(res, min_raw)
    print_fold_check(res["fold_check"])
    if compare:
        print_compare_old(res, cat, signal, min_raw)
    stem = f"{cat}_{signal}_2fold"
    save_json(res, DIAG_RESULTS_DIR / f"bkg_nn_hist_{stem}.json")
    if plots:
        DIAG_PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plot_yield_per_sample(res, DIAG_PLOT_DIR / f"{stem}_bkg_yield_per_sample.png", min_raw)
        plot_occupancy(res, DIAG_PLOT_DIR / f"{stem}_bkg_occupancy.png")
        plot_stack(res, DIAG_PLOT_DIR / f"{stem}_bkg_stack.png")
        print(f"\nsaved plots/diagnostics/{stem}_bkg_{{yield_per_sample,occupancy,stack}}.png and "
              f"results/diagnostics/bkg_nn_hist_{stem}.json")
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=[800],
                   help="signal masses whose model is evaluated (default 800)")
    p.add_argument("--bins", type=int, default=NN_BINS,
                   help=f"number of NN bins on [0,1] (default config.NN_BINS = {NN_BINS}, the limit's binning)")
    p.add_argument("--hi-cut", type=float, default=0.5,
                   help="score above which a sample counts as 'signal-like' in the table (default 0.5)")
    p.add_argument("--min-raw", type=int, default=10,
                   help="bins with fewer MC events than this are flagged in the plots / warnings (default 10)")
    p.add_argument("--no-plots", action="store_true", help="tables and json only")
    p.add_argument("--no-compare", action="store_true",
                   help="do not compare with the single-split json of the old version of this script")
    a = p.parse_args()

    from tttg.data.features import PrepareFeaturesForTraning
    from tttg.data.weights import WeightManager

    ds, wm = PrepareFeaturesForTraning(), WeightManager()
    for m in a.masses:
        run_one(ds, wm, a.cat, m, a.bins, a.hi_cut, a.min_raw, plots=not a.no_plots, compare=not a.no_compare)


if __name__ == "__main__":
    main()
