"""Plot the expected cross-section limits (fb) with the theory curve.

Command line examples (from the project root):
    python -m tttg.plotting.xsec_limits                              # all categories, one plot, with bands
    python -m tttg.plotting.xsec_limits --cats noEFT sEFT            # only these categories, one plot
    python -m tttg.plotting.xsec_limits --cats sEFT --no-bands       # one category, no uncertainty bands
    python -m tttg.plotting.xsec_limits --separate                   # one plot per category
    python -m tttg.plotting.xsec_limits --limits results/limits_combine.json # plot the Combine results instead
    python -m tttg.plotting.xsec_limits --show                       # also open the figures

From a notebook:
    plot_xsec_limits(limits, xsecs, cats=["noEFT", "sEFT"], combined=True, bands=False)
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tttg.config import CAT_LABELS, CATS, LIMITS_JSON, LIMIT_PLOT_DIR, MASSES
from tttg.limits.io import load_limits, load_signal_xsecs, xsec_limits
from tttg.plotting.style import CMS_COLORS, style_axes

BAND_NAMES = ("minus2", "minus1", "median", "plus1", "plus2")


def _check_cats(cats):
    cats = list(CATS) if cats is None else list(cats)
    unknown = [c for c in cats if c not in CATS]
    if unknown:
        raise ValueError(f"unknown categories {unknown}; choose from {CATS}")
    if not cats:
        raise ValueError("no categories selected")
    return cats


def _draw(ax, cats, bands_data, theory, bands):
    """Draw the selected categories on one axes."""
    masses = np.array(MASSES, dtype=float)
    first = True
    for cat in cats:
        color = CMS_COLORS[CATS.index(cat)]      # colour fixed per category, whatever is selected
        ok = np.isfinite(bands_data["median"][cat])
        if not ok.any():
            print(f"[xsec_limits] {cat}: no valid limits, skipped")
            continue
        x = masses[ok]
        if bands:
            # wider band first so the narrower one stays visible; label the bands only once
            ax.fill_between(x, bands_data["minus2"][cat][ok], bands_data["plus2"][cat][ok],
                            color="gold", alpha=0.4, label=r"$\pm 2\sigma$ expected" if first else None)
            ax.fill_between(x, bands_data["minus1"][cat][ok], bands_data["plus1"][cat][ok],
                            color="limegreen", alpha=0.5, label=r"$\pm 1\sigma$ expected" if first else None)
        ax.plot(x, bands_data["median"][cat][ok], "o-", color=color, linewidth=2, markersize=6,
                label=f"Expected limit: {CAT_LABELS[cat]}")
        first = False

    if theory is not None:
        ax.plot(masses, theory, "s-", color="red", linewidth=2, markersize=5,
                label=r"Theory ($g/\Lambda$ = 0.1 TeV$^{-1}$)")

    ax.set_xlabel("Mass (GeV)", fontsize=18)
    ax.set_ylabel(r"$\sigma_{pp\rightarrow tttg}$ (fb)", fontsize=18)
    ax.set_title(r"Limit on $pp\rightarrow tttg$", fontsize=20)
    ax.set_yscale("log")
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=12)


def plot_xsec_limits(limits, xsecs, cats=None, combined=True, bands=True, theory=True,
                     out=None, out_dir=LIMIT_PLOT_DIR, tag=""):
    """Plot expected cross-section limits.

    cats      categories to plot (default: all in config.CATS)
    combined  True: all selected categories in one figure; False: one figure per category
    bands     draw the +-1 / +-2 sigma expected bands
    theory    draw the theory cross section at g/Lambda = 0.1 TeV^-1
    out       explicit file name (only used when combined=True); by default a name is
              built from the selection, e.g. xsec_limit_noEFT-sEFT.png
    out_dir   folder for the saved plots
    tag       text appended to the default file names, e.g. "_combine"

    Returns the list of figures.
    """
    cats = _check_cats(cats)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = BAND_NAMES if bands else ("median",)
    bands_data = {b: xsec_limits(limits, xsecs, b, cats=cats) for b in names}
    # bands_data = {b: xsec_limits(limits, xsecs, b) for b in names}
    theory_vals = np.array([xsecs[m] for m in MASSES]) if theory else None
    suffix = ("" if bands else "_nobands") + tag

    groups = [cats] if combined else [[c] for c in cats]
    figs = []
    for group in groups:
        fig, ax = plt.subplots(figsize=(10, 8))
        _draw(ax, group, bands_data, theory_vals, bands)
        fig.tight_layout()
        path = out if (combined and out is not None) else out_dir / f"xsec_limit_{'-'.join(group)}{suffix}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[xsec_limits] saved {path}")
        figs.append(fig)
    return figs


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cats", nargs="+", choices=CATS, default=None,
                   help="categories to plot (default: all)")
    p.add_argument("--separate", action="store_true",
                   help="one plot per category instead of all in one plot")
    p.add_argument("--no-bands", action="store_true", help="do not draw the uncertainty bands")
    p.add_argument("--no-theory", action="store_true", help="do not draw the theory curve")
    p.add_argument("--limits", type=Path, default=LIMITS_JSON,
                   help="limits json to plot, e.g. results/limits_combine.json (default: results/limits_pyhf.json)")
    p.add_argument("--show", action="store_true", help="open the figures after saving")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tag = "" if args.limits == LIMITS_JSON else f"_{args.limits.stem}"
    plot_xsec_limits(load_limits(args.limits), load_signal_xsecs(), cats=args.cats,
                     combined=not args.separate, bands=not args.no_bands,
                     theory=not args.no_theory, tag=tag)
    if args.show:
        plt.show()
