"""Loading limit json files / signal cross sections and turning mu limits into cross-section limits."""
import json

import numpy as np

from tttg.config import BAND_INDEX, CATS, COFFEA_OUTPUT, LIMITS_JSON, MASSES


def load_limits(path=LIMITS_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def load_signal_xsecs(path=COFFEA_OUTPUT) -> dict[int, float]:
    """Signal cross sections [fb] at the reference coupling, from the coffea metadata."""
    from coffea.util import load  # imported lazily so the other modules work without coffea

    fileset = load(str(path))["metadata"]
    return {m: fileset[f"Signal_{m}"]["xsec"] for m in MASSES}


def xsec_limits(limits: dict, xsecs: dict[int, float], band: str = "median", cats=None) -> dict[str, np.ndarray]:
    """Expected cross-section limit (mu limit * theory xsec) per category, ordered as MASSES.

    Masses whose fit failed in limits/pyhf_limits.py come back as NaN.
    """
    k = BAND_INDEX[band]
    cats = CATS if cats is None else cats
    out = {}
    for cat in cats:
        vals = []
        for m in MASSES:
            entry = limits[cat].get(str(m))
            vals.append(np.nan if entry is None else entry["expected"][k] * xsecs[m])
        out[cat] = np.array(vals, dtype=float)
    return out
