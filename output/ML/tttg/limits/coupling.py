"""Limit on g/Lambda from the median cross-section limit.

For every mass:
  1. fit  sigma(g) = a4 g^4 + a5 g^5 + a6 g^6 + a7 g^7 + a8 g^8  to tttg_CrossSection.txt
  2. solve sigma(g) = median cross-section limit  (smallest positive real root)
  3. g/Lambda = g / Lambda   (Lambda = 5 TeV)
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from tttg.config import (CATS, G_LIMITS_JSON, LAMBDA_TEV, MASSES, XSEC_G_COLUMN, XSEC_MASS_COLUMN,
                         XSEC_TABLE, XSEC_UNIT_FACTOR, XSEC_VALUE_COLUMN, CAT_LABELS, LIMIT_PLOT_DIR)
from tttg.limits.io import load_limits, load_signal_xsecs, xsec_limits
from tttg.plotting.g_limits import plot_all, plot_g_limit

POWERS = (4, 5, 6, 7, 8)


def sigma_model(g, a4, a5, a6, a7, a8):
    return a4 * g**4 + a5 * g**5 + a6 * g**6 + a7 * g**7 + a8 * g**8


def fit_coefficients(path=XSEC_TABLE) -> dict[float, np.ndarray]:
    """mass -> [a4, a5, a6, a7, a8], with sigma in fb."""
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.lstrip("#") for c in df.columns]
    coef = {}
    for mass, grp in df.groupby(XSEC_MASS_COLUMN):
        g = grp[XSEC_G_COLUMN].to_numpy(float)
        sigma = grp[XSEC_VALUE_COLUMN].to_numpy(float) * XSEC_UNIT_FACTOR
        coef[float(mass)], _ = curve_fit(sigma_model, g, sigma, p0=np.ones(5))
    return coef


def solve_g(coef, sigma_limit) -> float:
    """Smallest positive real g with sigma(g) = sigma_limit (NaN if there is none)."""
    if not np.isfinite(sigma_limit) or sigma_limit <= 0:
        return np.nan
    a4, a5, a6, a7, a8 = coef
    # polynomial in g, highest power first: g^8 ... g^4, g^3, g^2, g^1, constant
    roots = np.roots([a8, a7, a6, a5, a4, 0, 0, 0, -sigma_limit])
    real = roots[np.abs(roots.imag) < 1e-8 * np.maximum(1, np.abs(roots.real))].real
    real = real[real > 0]
    return float(real.min()) if len(real) else np.nan


def g_over_lambda_limits(coef, sigma_lims) -> dict[str, dict[int, float]]:
    """cat -> {mass: limit on g/Lambda in TeV^-1}."""
    return {cat: {m: solve_g(coef[float(m)], s) / LAMBDA_TEV for m, s in zip(MASSES, sigma_lims[cat])}
            for cat in CATS}


def main(show=False):
    # cross-section limit = mu limit * theory cross section (needs the coffea xsec)
    sigma_lims = xsec_limits(load_limits(), load_signal_xsecs(), "median")
    limits = g_over_lambda_limits(fit_coefficients(), sigma_lims)

    G_LIMITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(G_LIMITS_JSON, "w") as f:
        json.dump({c: {str(m): (None if np.isnan(v) else v) for m, v in d.items()}
                   for c, d in limits.items()}, f, indent=1)

    LIMIT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for cat in CATS:
        print(cat, {m: round(v, 3) for m, v in limits[cat].items()})
        plot_g_limit(limits[cat], CAT_LABELS[cat], out=LIMIT_PLOT_DIR / f"g_limit_{cat}.png")
    plot_all(limits)
    if show:
        plt.show()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="open the figures after saving")
    main(show=ap.parse_args().show)