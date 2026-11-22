"""pyhf limits from the experimental models of weighted_training.py, WITHOUT touching models/ or results/limits*.json.

    python -m diagnostics.experiment_limits --cat noEFT --masses 500 800 1000 1500
    python -m diagnostics.experiment_limits --masses 800 --sources models baseline proc:0.5

Run from the project root. For every mass and every SOURCE it runs the same chain as
python -m tttg.limits.pyhf_limits (NN histograms -> rebin -> pyhf model -> CLs scan), using your CURRENT
tttg/limits/pyhf_limits.py (rebin, make_bkgs, extract_limit), so a change there (e.g. staterror) is picked up.
The only trick: for the duration of the call the model file that tttg/histograms/nn_histograms.py loads is
swapped in memory; nothing is copied or overwritten on disk.

Sources (default: models baseline proc:0.5 - the models you have now, the retrained baseline and the weighted one)
  models          models/<cat>/Signal_<m>.pth            (the models used for the analysis now)
  <scheme>        every models_experiments/<scheme>/<cat>/Signal_<m>_seed<k>.pth, one limit per seed
                  (e.g. baseline, proc:0.5, proc:1)

Prints, per mass, the expected median limit on mu (and the +-1 sigma band) for every source/seed, the mean +- std
over seeds, and the ratio to the baseline; then (section B) where the processes that were DROPPED from the weighted
loss (few events) end up in the NN score and what the background looks like at high score.
Results are cached in results/diagnostics/experiment_limits_<cat>.json (use --force to recompute).
mu limits are in units of the signal cross section assumed in the histograms, exactly as in limits_pyhf.json.
"""
import argparse
import contextlib
import json
import re

import numpy as np

from tttg.config import BACKGROUNDS, CATS, MASSES, MODEL_DIR, RESULTS_DIR, model_path

EXP_DIR = MODEL_DIR.parent / "models_experiments"
OUT_DIR = RESULTS_DIR / "diagnostics"
FEW = 100                      # a process with fewer test events than this is "few-event"


# ----------------------------------------------------------------------------- sources
def find_models(source, cat, mass):
    """[(label, path)] for one source."""
    if source == "models":
        p = model_path(cat, mass)
        return [("models", p)] if p.exists() else []
    d = EXP_DIR / source.replace(":", "_") / cat
    found = []
    for p in d.glob(f"Signal_{mass}_seed*.pth"):
        k = int(re.search(r"seed(\d+)", p.name).group(1))
        found.append((k, p))
    return [(f"{source}/seed{k}", p) for k, p in sorted(found)]


@contextlib.contextmanager
def use_model(path):
    """nn_histograms loads model_path(cat, m): point it at `path` for the duration of the block (in memory only)."""
    from tttg.histograms import nn_histograms as nh
    old = nh.model_path
    nh.model_path = lambda cat, m: path
    try:
        yield
    finally:
        nh.model_path = old


# ----------------------------------------------------------------------------- limit
def limit_for(tws, mass, cat, path, grouping, use_sig_stat):
    """Same body as the loop in tttg.limits.pyhf_limits.run, for one mass and one model file."""
    from tttg.histograms.extractor import ExtractHistData
    from tttg.histograms.nn_histograms import get_histograms
    from tttg.limits import pyhf_limits as P

    name = f"Signal_{mass}"
    with use_model(path):
        hists = get_histograms(mass, tws, cat)
    hd = ExtractHistData(hists).extract_hist_info(MC_BCs=BACKGROUNDS, signals=[name])
    S, B, V, rel, groups = P.rebin(hd.signals[name], hd.total_backgrounds, hd.bkg_var, hd.syst[name])
    Vs = P.merge_groups(hd.signals_var[name], groups)
    bkgs = P.make_bkgs(hd, groups, grouping)
    res = P.extract_limit(S, bkgs, rel, Vs=Vs, use_sig_syst=True, use_sig_stat=use_sig_stat)
    res.pop("total", None)                         # the full CLs scan is not needed here
    res["n_bins"] = len(groups)
    return res


# ----------------------------------------------------------------------------- tables
def summarise(results):
    """results: {label: limit dict} for one mass -> {source: dict(median, lo, hi, n, per_seed)}"""
    by_src = {}
    for label, r in results.items():
        by_src.setdefault(label.split("/")[0], []).append(r)
    out = {}
    for src, rs in by_src.items():
        med = np.array([r["expected"][2] for r in rs], float)
        lo = np.array([r["expected"][1] for r in rs], float)
        hi = np.array([r["expected"][3] for r in rs], float)
        out[src] = dict(median=float(np.nanmean(med)), std=float(np.nanstd(med)) if len(med) > 1 else 0.0,
                        lo=float(np.nanmean(lo)), hi=float(np.nanmean(hi)), n=len(rs),
                        per_seed=[float(x) for x in med], bins=[r.get("n_bins") for r in rs])
    return out


def print_mass(mass, results, baseline="baseline", out=print):
    out(f"\n  Signal_{mass}: expected limit on mu (median; band = -1/+1 sigma)")
    for label, r in results.items():
        e = r["expected"]
        out(f"    {label:<18} {e[2]:9.3f}   [{e[1]:8.3f}, {e[3]:8.3f}]   bins after rebin: {r.get('n_bins')}")
    s = summarise(results)
    ref = s.get(baseline, {}).get("median")
    out(f"    {'source':<14}{'n':>3}{'mean +- std':>20}{'vs ' + baseline:>14}")
    for src, v in s.items():
        ratio = f"{v['median'] / ref:>13.2f}x" if ref and np.isfinite(ref) and ref > 0 else f"{'-':>14}"
        out(f"    {src:<14}{v['n']:>3}{v['median']:>12.3f} +- {v['std']:<6.3f}{ratio}")
    return s


# ----------------------------------------------------------------------------- B: where do the few-event processes go
def scores_for(tws, path):
    import torch
    from tttg.training.model import OptimizedHEPClassifier
    model = OptimizedHEPClassifier(next(iter(tws.values())).shape[1])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        return {s: torch.sigmoid(model(torch.FloatTensor(x))).numpy().ravel() for s, x in tws.items()}


def composition_report(scores, weights, signal, label, thresholds=(0.5, 0.8), out=print, n_show=20):
    bkgs = [b for b in scores if b != signal]
    few = [b for b in bkgs if len(scores[b]) < FEW]
    out(f"\n  [{label}]")
    out(f"    few-event processes: score of every test event (sorted, high first), weight/event")
    for b in sorted(few, key=lambda b: -weights[b].sum()):
        sc = np.sort(scores[b])[::-1][:n_show]
        out(f"      {b:<8} n={len(scores[b]):>3}  w/ev={weights[b][0]:9.4g}  scores: " + " ".join(f"{x:.2f}" for x in sc))
    for t in thresholds:
        parts = {b: weights[b][scores[b] > t].sum() for b in bkgs}
        raw = {b: int((scores[b] > t).sum()) for b in bkgs}
        tot = sum(parts.values())
        sig = weights[signal][scores[signal] > t].sum()
        out(f"    score > {t}: S={sig:.1f}  B={tot:.1f}  S/sqrt(B)={sig / np.sqrt(tot) if tot > 0 else np.nan:.2f};  background composition:")
        for b in sorted(bkgs, key=lambda b: -parts[b])[:6]:
            if tot > 0 and parts[b] > 0.005 * tot:
                mark = "  <-- few-event process" if b in few else ""
                out(f"        {b:<8}{100 * parts[b] / tot:6.1f}%   ({raw[b]} raw events){mark}")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=[500, 800, 1000, 1500, 2000])
    p.add_argument("--sources", nargs="+", default=["models", "baseline", "proc:0.5"])
    p.add_argument("--bkg-grouping", choices=["total", "groups"], default="total")
    p.add_argument("--no-sig-stat", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p.add_argument("--no-composition", action="store_true", help="skip section B")
    a = p.parse_args()

    from tttg.data.features import PrepareFeaturesForTraning
    ds = PrepareFeaturesForTraning()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = OUT_DIR / f"experiment_limits_{a.cat}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    opts = f"{a.bkg_grouping}|sigstat={not a.no_sig_stat}"

    print(f"limit options: grouping={a.bkg_grouping}, signal MC stat={'off' if a.no_sig_stat else 'on'}")
    for m in a.masses:
        tws = ds.get_test_with_sample(signal=f"Signal_{m}", cat=a.cat)
        results, paths = {}, {}
        for src in a.sources:
            found = find_models(src, a.cat, m)
            if not found:
                print(f"  (no models for source '{src}' at mass {m} - skipped)")
            for label, path in found:
                paths[label] = path
                key = f"{opts}|{label}|{m}"
                if key in cache and not a.force:
                    results[label] = cache[key]
                    continue
                print(f"  computing limit: {label}, mass {m} ...", flush=True)
                try:
                    results[label] = limit_for(tws, m, a.cat, path, a.bkg_grouping, not a.no_sig_stat)
                except Exception as e:                       # keep going, report at the end
                    print(f"    FAILED: {e}")
                    continue
                cache[key] = results[label]
                cache_file.write_text(json.dumps(cache, indent=1))
        if results:
            print_mass(m, results)

        if not a.no_composition and paths:
            from diagnostics.common import load_weights
            print(f"\n  B. Signal_{m}: few-event processes and background composition at high score")
            weights = None
            for label in [l for l in paths if l == "models" or l.endswith("/seed0")]:
                sc = scores_for(tws, paths[label])
                if weights is None:
                    weights = load_weights(a.cat, sc)
                composition_report(sc, weights, f"Signal_{m}", label)
    print(f"\ncached in {cache_file}")


if __name__ == "__main__":
    main()