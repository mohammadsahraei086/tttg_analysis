"""Expected 95% CL limits on the signal strength mu (pyhf) -> results/limits_pyhf.json (one entry per cat/mass, updated in place).

Per category and signal mass:
  NN-output histograms -> rebin (bkg MC stat) -> pyhf model -> CLs scan -> mu limits
"""
import fcntl
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pyhf

from tttg.config import (BACKGROUNDS, BKG_REL_UNC, BKG_REL_UNC_BY_GROUP, CATS, LIMITS_JSON, MASSES,
                         RESULTS_DIR, SIG_SYSTS)
from tttg.data.features import PrepareFeaturesForTraning
from tttg.histograms.extractor import ExtractHistData
from tttg.histograms.nn_histograms import get_histograms


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------
def rebin(sig, bkg, bvar, syst, max_stat=0.3):
    """Merge bins from the high-NN end until the bkg MC stat error is <= max_stat."""
    sig, bkg, bvar = (np.asarray(x, float) for x in (sig, bkg, bvar))
    groups, cur = [], []
    for i in range(len(bkg) - 1, -1, -1):
        cur.append(i)
        b, v = bkg[cur].sum(), bvar[cur].sum()
        if b > 0 and np.sqrt(v) / b <= max_stat:
            groups.append(sorted(cur))
            cur = []
    if cur:  # leftover low-NN bins join the lowest group
        groups[-1:] = [sorted(cur) + groups[-1]] if groups else [sorted(cur)]
    groups = groups[::-1]

    S = np.array([sig[g].sum() for g in groups])
    B = np.array([bkg[g].sum() for g in groups])
    V = np.array([bvar[g].sum() for g in groups])

    # signal syst: yield-weighted relative shift per merged bin (fully correlated, adds linearly)
    rel = {}
    for n in SIG_SYSTS:
        r = np.nan_to_num(np.asarray(syst[f"{n}_unc"], float), nan=0.0, posinf=0.0)
        rel[n] = np.array(
            [(r[g] * sig[g]).sum() / sig[g].sum() if sig[g].sum() > 0 else 0.0 for g in groups]
        )
    return S, B, V, rel, groups


def bin_merger(edges, sig, bkg, bvar, syst, max_stat=0.3):
    """Same bin merging as rebin(), for writing merged histograms.

    Returns (new_edges, merge); merge(a) sums any per-bin array `a` over the merged bins.
    """
    *_, groups = rebin(sig, bkg, bvar, syst, max_stat)
    new_edges = np.asarray(edges, float)[[g[0] for g in groups] + [len(edges) - 1]]

    def merge(a):
        a = np.asarray(a, float)
        return np.array([a[g].sum() for g in groups])

    return new_edges, merge


def merge_groups(a, groups):
    """Sum a per-bin array over the merged bins returned by rebin()."""
    a = np.asarray(a, float)
    return np.array([a[g].sum() for g in groups])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _safe(name):
    """pyhf-friendly modifier name (group names contain '+')."""
    return re.sub(r"\W", "_", name)


def bkg_unc_for(name):
    """Relative normalisation uncertainty of a background group (config.BKG_REL_UNC[_BY_GROUP])."""
    return BKG_REL_UNC_BY_GROUP.get(name, BKG_REL_UNC)


def build_model(S, bkgs, rel, Vs=None, bkg_rel_unc=None,
                use_sig_syst=True, use_shape=True, use_sig_stat=True, use_bkg_stat=True):
    S = np.clip(S, 0, None)
    mods = [{"name": "mu", "type": "normfactor", "data": None}]

    if use_sig_syst:
        for n in SIG_SYSTS:
            r = np.clip(rel[n], 0, 0.9)                          # per-bin relative shift
            rbar = float((r * S).sum() / max(S.sum(), 1e-12))    # yield-weighted mean = norm part
            mods.append({"name": f"sig_{n}", "type": "normsys",
                         "data": {"hi": 1 + rbar, "lo": 1 - rbar}})
            if use_shape:                                        # same name -> same nuisance parameter
                mods.append({"name": f"sig_{n}", "type": "histosys",
                             "data": {"hi_data": (S * (1 + r) / (1 + rbar)).tolist(),
                                      "lo_data": (S * (1 - r) / (1 - rbar)).tolist()}})

    if use_sig_stat and Vs is not None:      # signal MC stat -> shared "mcstat" gamma
        sig_err = np.where(S > 0, np.sqrt(np.clip(Vs, 0, None)), 0.0)
        mods.append({"name": "mcstat", "type": "staterror", "data": sig_err.tolist()})

    samples = [{"name": "signal", "data": S.tolist(), "modifiers": mods}]
    B_total = np.zeros_like(S, dtype=float)
    for name, (B, V) in bkgs.items():
        B = np.clip(B, 1e-6, None)
        B_total = B_total + B
        u = bkg_unc_for(name) if bkg_rel_unc is None else (
            bkg_rel_unc.get(name, BKG_REL_UNC) if isinstance(bkg_rel_unc, dict) else bkg_rel_unc)
        bmods = [
            # one normalisation nuisance shared by all background samples (fully correlated)
            {"name": "bkg_norm", "type": "normsys", "data": {"hi": 1 + u, "lo": 1 - u}},
        ]
        if use_bkg_stat:    # --no-bkg-stat switches the background MC-statistical uncertainty off
            # same name as the signal's -> one gamma per bin for ALL samples (Barlow-Beeston-lite)
            bmods.append({"name": "mcstat", "type": "staterror", "data": np.sqrt(V).tolist()})
        samples.append({"name": f"bkg_{_safe(name)}", "data": B.tolist(), "modifiers": bmods})

    spec = {"channels": [{"name": "SR", "samples": samples}]}
    return pyhf.Model(spec, poi_name="mu"), B_total

# ---------------------------------------------------------------------------
# CLs scan
# ---------------------------------------------------------------------------
def _crossing(mus, cls, level=0.05):
    """mu at which CLs crosses `level` (log-log interpolation).

    Returns 0.0 if already excluded at the first scan point, inf if never excluded.
    """
    mus, cls = np.asarray(mus, float), np.asarray(cls, float)
    ok = np.isfinite(cls) & (cls > 0)             # drop NaN and failed (CLs == 0) points
    mus, cls = mus[ok], cls[ok]
    if len(cls) == 0:
        return np.nan
    above = np.where(cls >= level)[0]
    if len(above) == 0:
        return 0.0
    if above[-1] == len(cls) - 1:
        return np.inf
    i = above[-1] + 1                             # first point after the LAST point with CLs >= level
    x0, x1 = np.log(mus[i - 1]), np.log(mus[i])
    y0, y1 = np.log(cls[i - 1]), np.log(cls[i])
    return float(np.exp(x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0)))


def extract_limit(S, bkgs, rel, Vs=None, mu_lo=0.01, mu_hi=100, n_scan=100, **model_kw):
    model, B = build_model(S, bkgs, rel, Vs=Vs, **model_kw)
    data = list(B) + model.config.auxdata                    # Asimov, background-only
    bounds = model.config.suggested_bounds()
    bounds[model.config.poi_index] = (0, mu_hi)

    obs, exp, ok_mus = [], [], []
    for mu in np.geomspace(mu_lo, mu_hi, n_scan):
        try:
            o, e = pyhf.infer.hypotest(mu, data, model, test_stat="qtilde",
                                       return_expected_set=True, par_bounds=bounds)
        except pyhf.exceptions.FailedMinimization:
            continue                                         # skip this point, keep the mass
        obs.append(float(o))
        exp.append([float(x) for x in e])
        ok_mus.append(mu)
        if max(e) < 0.05:                                    # +2 sigma band excluded -> all bands done
            break
    mus, exp = np.array(ok_mus), np.array(exp)

    return {"observed": _crossing(mus, np.array(obs)),
            "expected": [_crossing(mus, exp[:, k]) for k in range(5)],  # -2s, -1s, median, +1s, +2s
            "total": [mus.tolist(), exp.tolist()],
            "n_nan": int(np.sum(~np.isfinite(exp[:, 2])))}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def make_bkgs(hd, groups, grouping):
    """{name: (B, V)} per bin after merging.

    grouping="total"   all backgrounds summed into one sample
    grouping="groups"  one sample per BACKGROUNDS group (config.BACKGROUNDS)
    """
    if grouping == "total":
        return {"background": (merge_groups(hd.total_backgrounds, groups),
                               merge_groups(hd.bkg_var, groups))}
    bkgs = {}
    for name in hd.backgrounds:
        B = merge_groups(hd.backgrounds[name], groups)
        if B.sum() > 0:                      # skip empty groups
            bkgs[name] = (B, merge_groups(hd.backgrounds_var[name], groups))
    return bkgs


def diagnose(S, B, V):
    for i in range(len(S)):
        mc = np.sqrt(V[i]) / B[i] if B[i] > 0 else np.inf
        print(f"   bin{i:2d}  S={S[i]:9.3g}  B={B[i]:9.3g}  MCstat={mc:5.2f}  S/B={S[i] / max(B[i], 1e-9):8.2g}")


# ---------------------------------------------------------------------------
# saving: one json, updated entry by entry
# ---------------------------------------------------------------------------
def _read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        # a half-written / damaged file: keep it for inspection instead of silently overwriting it
        backup = path.with_suffix(".json.corrupt")
        path.replace(backup)
        print(f"WARNING: {path} was not valid json, moved to {backup}; starting a new file")
        return {}


def save_limit(out, cat, m, res):
    """Insert/replace ONE (cat, mass) entry in the limits json and leave every other entry untouched.

    Read-modify-write under an exclusive file lock (so several processes, e.g. one per category,
    can write the same file) and written through a temporary file + os.replace (so a crash while
    writing can never leave a half-written json).
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_name(out.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            limits = _read_json(out)
            limits.setdefault(cat, {})[str(m)] = res          # json keys are strings (load_limits uses str(m))
            tmp = out.with_name(out.name + f".tmp{os.getpid()}")
            with open(tmp, "w") as f:
                json.dump(limits, f, indent=1)
            os.replace(tmp, out)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(out, verbose=False, grouping="total", use_sig_stat=True, cats=None, masses=None,
        skip_existing=False, **model_kw):
    """Compute the limits of the requested (cat, mass) pairs; each result is written to `out` as soon as
    it is ready. A failure in one pair is reported and skipped; entries already in the file are never lost."""
    cats = list(cats) if cats else list(CATS)
    masses = list(masses) if masses else list(MASSES)
    dataset = PrepareFeaturesForTraning()
    done = _read_json(out)
    n_ok, failed = 0, []
    for cat in cats:
        for m in masses:
            if skip_existing and str(m) in done.get(cat, {}):
                print(f"{cat} {m}: already in {Path(out).name}, skipped")
                continue
            name = f"Signal_{m}"
            try:
                t0 = time.perf_counter()
                hists = get_histograms(m, dataset.get_test_with_sample(signal=name, cat=cat), cat)
                hd = ExtractHistData(hists).extract_hist_info(MC_BCs=BACKGROUNDS, signals=[name])
                t1 = time.perf_counter()

                # bin merging is always decided on the TOTAL background, whatever the grouping
                S, B, V, rel, groups = rebin(hd.signals[name], hd.total_backgrounds,
                                             hd.bkg_var, hd.syst[name])
                Vs = merge_groups(hd.signals_var[name], groups)
                bkgs = make_bkgs(hd, groups, grouping)
                print(f"{cat} {m}: {len(groups)} bins after rebin, {len(bkgs)} background sample(s)")
                if verbose:
                    diagnose(S, B, V)
                    print("signal MC stat:", np.round(np.sqrt(Vs) / np.maximum(S, 1e-12), 3))
                    for n in SIG_SYSTS:
                        print(n, np.round(rel[n], 3))
                    print("total:", np.sqrt(sum(rel[n] ** 2 for n in SIG_SYSTS)).round(3))

                t2 = time.perf_counter()
                res = extract_limit(S, bkgs, rel, Vs=Vs, use_sig_syst=True,
                                    use_sig_stat=use_sig_stat, **model_kw)
                t3 = time.perf_counter()
            except Exception as e:                     # one bad (cat, mass) must not stop the others
                print(f"   FAILED {cat} {m}: {type(e).__name__}: {e}")
                failed.append((cat, m))
                continue
            save_limit(out, cat, m, res)
            n_ok += 1
            print(f"   exp median = {res['expected']}  (NaN points: {res['n_nan']})")
            print(f"   saved {cat}/{m} -> {out}   time: histograms {t1 - t0:.1f}s, "
                  f"pyhf scan {t3 - t2:.1f}s ({len(res['total'][0])} mu points)")
    print(f"\n{n_ok} limit(s) written" + (f", {len(failed)} failed: {failed}" if failed else ""))


def main(verbose=False, out=None, grouping="total", **kw):
    if out is None:
        out = LIMITS_JSON if grouping == "total" else RESULTS_DIR / f"limits_pyhf_{grouping}.json"
        if not kw.get("use_bkg_stat", True):      # never overwrite the normal limits with a test configuration
            out = Path(out).with_name(Path(out).stem + "_nobkgstat.json")
    run(Path(out), verbose=verbose, grouping=grouping, **kw)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bkg-grouping", choices=["total", "groups"], default="total",
                    help="total (default): all backgrounds summed into one pyhf sample. "
                         "groups: one pyhf sample per group in config.BACKGROUNDS")
    ap.add_argument("--no-sig-stat", action="store_true",
                    help="do not include the signal MC-statistical uncertainty")
    ap.add_argument("--no-bkg-stat", action="store_true",
                    help="do not include the background MC-statistical uncertainty (a TEST: shows how much the limit "
                         "is driven by it). Default output: results/limits_pyhf[_groups]_nobkgstat.json")
    ap.add_argument("--cats", nargs="+", choices=CATS, default=None, help="categories to run (default: all)")
    ap.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=None,
                    help="signal masses to run (default: all)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not recompute (cat, mass) pairs that are already in the json (resume after a crash)")
    ap.add_argument("--mu-lo", type=float, default=0.01, help="first mu of the CLs scan (default 0.01)")
    ap.add_argument("--mu-hi", type=float, default=100, help="last mu of the CLs scan (default 100)")
    ap.add_argument("--n-scan", type=int, default=100, help="number of mu points of the scan (default 100)")
    ap.add_argument("--out", default=None,
                    help="output json (default: results/limits_pyhf.json for --bkg-grouping total, "
                         "results/limits_pyhf_groups.json for groups). Entries of other (cat, mass) pairs "
                         "already in the file are kept.")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    main(verbose=a.verbose, out=a.out, grouping=a.bkg_grouping, use_sig_stat=not a.no_sig_stat, use_bkg_stat=not a.no_bkg_stat,
         cats=a.cats, masses=a.masses, skip_existing=a.skip_existing,
         mu_lo=a.mu_lo, mu_hi=a.mu_hi, n_scan=a.n_scan)