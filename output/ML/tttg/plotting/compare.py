"""Compare the expected limits on mu from pyhf (results/limits_pyhf.json) and Combine (results/limits_combine.json).

    python -m tttg.plotting.compare
    python -m tttg.plotting.compare --pyhf results/limits_pyhf_norm.json --cats noEFT
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from tttg.config import (BAND_INDEX, CAT_LABELS, CATS, LIMITS_COMBINE_JSON, LIMITS_JSON,
                         LIMIT_PLOT_DIR, MASSES)
from tttg.limits.io import load_limits
from tttg.plotting.style import style_axes


def median_mu(limits, cat):
    """Median expected mu limit per mass (NaN where missing or the fit failed)."""
    out = []
    for m in MASSES:
        entry = limits.get(cat, {}).get(str(m))
        v = np.nan if entry is None else entry["expected"][BAND_INDEX["median"]]
        out.append(v if np.isfinite(v) else np.nan)
    return np.array(out, dtype=float)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pyhf", default=LIMITS_JSON)
    p.add_argument("--combine", default=LIMITS_COMBINE_JSON)
    p.add_argument("--cats", nargs="+", choices=CATS, default=CATS)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    pyhf_lim, comb_lim = load_limits(args.pyhf), load_limits(args.combine)
    masses = np.array(MASSES, float)

    fig, axes = plt.subplots(2, len(args.cats), figsize=(6 * len(args.cats), 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]}, squeeze=False)
    print(f"{'cat':10}{'mass':>6}{'pyhf':>10}{'combine':>10}{'combine/pyhf':>14}")
    for j, cat in enumerate(args.cats):
        a, c = median_mu(pyhf_lim, cat), median_mu(comb_lim, cat)
        ratio = c / a
        for m, x, y, r in zip(MASSES, a, c, ratio):
            print(f"{cat:10}{m:>6}{x:>10.3g}{y:>10.3g}{r:>14.3f}")

        top, bot = axes[0, j], axes[1, j]
        top.plot(masses, a, "o--", color="#3f90da", label="pyhf")
        top.plot(masses, c, "s-", color="#bd1f01", label="Combine")
        top.set_yscale("log")
        top.set_ylabel(r"expected 95% CL limit on $\mu$", fontsize=14)
        top.set_title(CAT_LABELS[cat], fontsize=16)
        top.legend(fontsize=12)
        bot.plot(masses, ratio, "o-", color="black")
        bot.axhline(1, color="gray", lw=1)
        bot.set_ylabel("Combine / pyhf", fontsize=12)
        bot.set_xlabel("Mass (GeV)", fontsize=14)
        style_axes(top)
        style_axes(bot)

    fig.tight_layout()
    LIMIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = LIMIT_PLOT_DIR / "compare_pyhf_combine.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
