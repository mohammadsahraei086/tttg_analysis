"""Find out what produces the spikes in the NN-output distributions.

    python -m diagnostics.score_spikes --cat noEFT --mass 800
    python -m diagnostics.score_spikes --cat noEFT --mass 800 --bkg-range 0 0.02 --sig-range 0.74 0.82

For one (category, mass) it uses the trained model in models/ and the TEST half of every sample and prints

  1. event counts per sample (train / test)
  2. the most populated score bins for signal and background
  3. which samples populate the background / signal spike
  4. which input columns are different for the spike events (standardised mean shift, fraction of
     sentinel (<= -90) and zero values inside vs outside the spike, number of distinct values)
  5. duplicated rows inside the test set and rows shared between the train and test halves
     (leakage), per sample
and saves plots/diagnostics/<cat>_Signal_<mass>_{scores,drivers}.png
(run from the project root; this folder only READS from tttg/)
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from diagnostics.common import load_scores
from tttg.config import CATS, MASSES, PLOT_DIR

DIAG_PLOT_DIR = PLOT_DIR / "diagnostics"


# ----------------------------------------------------------------------------- helpers
def column_names(output, cat, signal, features):
    """Names of the columns of the feature matrix (array-valued features expand to name_0, name_1...)."""
    names = []
    for var in features:
        first = output["features"][cat][signal][var].value[0]
        if isinstance(first, np.ndarray):
            names += [f"{var}_{i}" for i in range(len(first))]
        else:
            names.append(var)
    return names


def row_hash(X):
    """One float per row; identical rows give identical values, different rows (almost) never do."""
    rnd = np.random.default_rng(0).standard_normal(X.shape[1])
    return X.astype(np.float64) @ rnd


def top_bins(scores, nbins=100, k=5):
    counts, edges = np.histogram(scores, bins=nbins, range=(0, 1))
    return [(edges[i], edges[i + 1], counts[i]) for i in np.argsort(counts)[::-1][:k]]


def in_range(s, lo, hi):
    return (s >= lo) & (s <= hi)


def column_drivers(X, mask, names, top=8):
    """Columns in which the events with mask=True differ most from the others (same class)."""
    R, rest = X[mask], X[~mask]
    if len(R) == 0 or len(rest) == 0:
        return []
    sd = rest.std(0)
    sd[sd == 0] = 1.0
    shift = (R.mean(0) - rest.mean(0)) / sd
    sub = X[:: max(1, len(X) // 200000)]
    rows = []
    for j in np.argsort(-np.abs(shift))[:top]:
        rows.append(dict(
            col=names[j], shift=shift[j],
            sentinel_in=np.mean(R[:, j] <= -90), sentinel_out=np.mean(rest[:, j] <= -90),
            zero_in=np.mean(R[:, j] == 0), zero_out=np.mean(rest[:, j] == 0),
            n_unique=len(np.unique(sub[:, j])), j=j))
    return rows


def print_drivers(title, rows):
    print(f"\n  {title}")
    print(f"  {'column':<22}{'shift[sigma]':>13}{'sentinel in/out':>18}{'zero in/out':>16}{'#unique':>9}")
    for r in rows:
        print(f"  {r['col']:<22}{r['shift']:>13.2f}{r['sentinel_in']:>9.2f}/{r['sentinel_out']:<8.2f}"
              f"{r['zero_in']:>8.2f}/{r['zero_out']:<7.2f}{r['n_unique']:>9d}")


# ----------------------------------------------------------------------------- analysis
def analyse(raw_test, scores, names, signal, bkg_range, sig_range, raw_train=None,
            out_prefix=None):
    bkgs = [s for s in raw_test if s != signal]

    print("\n== 1. events per sample (train / test)")
    for s in [signal] + bkgs:
        ntr = len(raw_train[s]) if raw_train else -1
        print(f"  {s:<14} {ntr:>9d} / {len(raw_test[s]):>9d}")

    s_sig = scores[signal]
    s_bkg = np.concatenate([scores[b] for b in bkgs])
    print("\n== 2. most populated 0.01-wide score bins")
    for lab, s in (("signal", s_sig), ("background", s_bkg)):
        print(f"  {lab}: " + ", ".join(f"[{a:.2f},{b:.2f}]:{c}" for a, b, c in top_bins(s)))

    print(f"\n== 3. who populates the spikes  (background {bkg_range}, signal {sig_range})")
    print(f"  {'sample':<14}{'in bkg spike':>14}{'fraction':>10}{'in sig spike':>14}{'fraction':>10}")
    for s in bkgs + [signal]:
        nb = in_range(scores[s], *bkg_range).sum()
        ns = in_range(scores[s], *sig_range).sum()
        print(f"  {s:<14}{nb:>14d}{nb / len(scores[s]):>10.3f}{ns:>14d}{ns / len(scores[s]):>10.3f}")

    print("\n== 4. input columns that distinguish the spike events (test half)")
    X_bkg = np.vstack([raw_test[b] for b in bkgs])
    drv_b = column_drivers(X_bkg, in_range(s_bkg, *bkg_range), names)
    drv_s = column_drivers(raw_test[signal], in_range(s_sig, *sig_range), names)
    print_drivers("background spike vs rest of background", drv_b)
    print_drivers("signal spike vs rest of signal", drv_s)

    print("\n== 5. duplicated rows / train-test overlap")
    print(f"  {'sample':<14}{'dup. in test':>14}{'shared with train':>19}")
    for s in [signal] + bkgs:
        h = row_hash(raw_test[s])
        dup = len(h) - len(np.unique(h))
        shared = len(np.intersect1d(h, row_hash(raw_train[s]))) if raw_train else -1
        print(f"  {s:<14}{dup:>14d}{shared:>19d}")

    if out_prefix is not None:
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        # plot 1: score distribution per sample
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in bkgs:
            ax.hist(scores[s], bins=100, range=(0, 1), histtype="step", density=True, label=s)
        ax.hist(s_sig, bins=100, range=(0, 1), histtype="stepfilled", alpha=0.4, density=True,
                color="C1", label=signal)
        ax.set_yscale("log"); ax.set_xlabel("NN output score"); ax.set_ylabel("density")
        ax.legend(fontsize=7, ncol=2); fig.tight_layout()
        fig.savefig(f"{out_prefix}_scores.png", dpi=150); plt.close(fig)

        # plot 2: top-3 driver columns, spike vs rest
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for row, (lab, rows, X, s, rng) in enumerate((
                ("background", drv_b, X_bkg, s_bkg, bkg_range),
                ("signal", drv_s, raw_test[signal], s_sig, sig_range))):
            m = in_range(s, *rng)
            for ax, r in zip(axes[row], rows[:3]):
                lo, hi = np.percentile(X[:, r["j"]], [0.5, 99.5])
                if hi <= lo:
                    lo, hi = lo - 1, hi + 1
                kw = dict(bins=50, range=(lo, hi), density=True, histtype="step")
                ax.hist(X[m, r["j"]], label=f"{lab} in spike", **kw)
                ax.hist(X[~m, r["j"]], label=f"{lab} rest", **kw)
                ax.set_title(r["col"], fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_drivers.png", dpi=150); plt.close(fig)
        print(f"\nsaved {out_prefix}_scores.png and {out_prefix}_drivers.png")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--mass", type=int, choices=MASSES, default=800)
    p.add_argument("--bkg-range", nargs=2, type=float, default=[0.0, 0.02],
                   help="score range of the background spike (default 0 0.02)")
    p.add_argument("--sig-range", nargs=2, type=float, default=[0.74, 0.82],
                   help="score range of the signal spike (default 0.74 0.82)")
    a = p.parse_args()

    signal = f"Signal_{a.mass}"
    ds, scores = load_scores(a.cat, a.mass)
    raw_test, raw_train = ds.test, ds.train
    names = column_names(ds.output, a.cat, signal, ds.training_features)

    analyse(raw_test, scores, names, signal, tuple(a.bkg_range), tuple(a.sig_range),
            raw_train=raw_train, out_prefix=DIAG_PLOT_DIR / f"{a.cat}_{signal}")


if __name__ == "__main__":
    main()
