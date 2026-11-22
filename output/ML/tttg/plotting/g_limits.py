"""Plots of the g/Lambda limits."""
import matplotlib.pyplot as plt
import numpy as np

from tttg.config import CAT_LABELS, CATS, LIMIT_PLOT_DIR
from tttg.plotting.style import CMS_COLORS, style_axes


def plot_g_limit(g_lim, label=None, out=None):
    """One category: limit line and hatched excluded region above it."""
    x = np.array(list(g_lim.keys()), float)
    y = np.array(list(g_lim.values()), float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y, "o-", linewidth=2.5, markersize=10, color="black", markeredgewidth=2,
            label="Expected 95% CL limit" + (f": {label}" if label else ""))
    ax.fill_between(x, y, y.max() * 10, alpha=0.3, color="red", hatch="///", label="Excluded at 95% CL")
    ax.set_xlabel("Mass [GeV]", fontsize=18)
    ax.set_ylabel(r"$g/\Lambda$ [TeV$^{-1}$]", fontsize=18)
    ax.set_ylim(0, y.max() * 2)
    ax.set_xlim(x.min() - 50, x.max() + 50)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_all(limits, out=LIMIT_PLOT_DIR / "g_limit_all_cats.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cat in enumerate(CATS):
        x = np.array(list(limits[cat].keys()), float)
        y = np.array(list(limits[cat].values()), float)
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], "o-", linewidth=2.5, markersize=8, color=CMS_COLORS[i],
                label=f"Expected 95% CL: {CAT_LABELS[cat]}")
    ax.set_xlabel("Mass [GeV]", fontsize=18)
    ax.set_ylabel(r"$g/\Lambda$ [TeV$^{-1}$]", fontsize=18)
    ax.set_ylim(bottom=0)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
