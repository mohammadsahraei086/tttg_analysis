"""Yield +- statistical uncertainty per NN bin, read directly from ExtractHistData / HistInfo (the objects the limit uses).

    cd diagnostics
    python yield_stat_from_extractor.py --cat noEFT --masses 500
    python yield_stat_from_extractor.py --cat noEFT --masses 500 --groups     # config.BACKGROUNDS groups instead of single processes

HistInfo holds, per NN bin:  signals[name], signals_var[name]  /  backgrounds[group], backgrounds_var[group]  /
total_backgrounds, bkg_var.   The *_var arrays are the sum of squared weights, so stat. uncertainty = sqrt(var).
One 'group' per process is requested via MC_BCs = {process: [process]}.  Total bkg = the processes of config.BACKGROUNDS
(exactly what the limit sums).  Bins are the NN_BINS bins before pyhf's rebin().
Writes results/diagnostics/yield_stat_<cat>_Signal_<m>.csv

Second table (no binning, summed over all NN bins), one row per sample, to CHECK the uncertainty:
  N_test   events of the TEST half (from the feature arrays)           w/event  xsec*lumi*2/N_gen (WeightManager)
  N_bins   = yield / w : events that really are in a bin (score >= 1.0 or NaN are in no bin)
  yield    sum of the weights from the histogram                        stat     sqrt(sum of squared weights) from the histogram
  w*sqrt(N) the expected uncertainty of N events of equal weight w      sqrt(N)  the NAIVE count error, NOT in the same units
  sqrt(yield) the error of a real data count with that yield            check    stat / (w*sqrt(N_bins)), must be 1.000
Writes results/diagnostics/yield_stat_total_<cat>_Signal_<m>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tttg.config import BACKGROUNDS, CATS, MASSES, RESULTS_DIR
from tttg.data.features import PrepareFeaturesForTraning
from tttg.data.weights import WeightManager
from tttg.histograms.extractor import ExtractHistData
from tttg.histograms.nn_histograms import get_histograms


def sample_table(hd, scaled, wm, cat, signal, procs):
    """Rows (one per sample) with yield and stat summed over ALL bins, plus the independent expectation w*sqrt(N)."""
    group = {s: g for g, lst in BACKGROUNDS.items() for s in lst}
    rows = []
    for name in [signal] + list(procs):
        y, var = (hd.signals[name], hd.signals_var[name]) if name == signal else (hd.backgrounds[name], hd.backgrounds_var[name])
        y, var = float(np.sum(y)), float(np.sum(var))
        n_test = len(scaled[name])
        if name == signal:
            w = wm.get_weights(name, "xsec", "luminosity", "sum_genweight", "variation", cat=cat, variation="Nominal")
        else:
            w = wm.get_weights(name, "xsec", "luminosity", "sum_genweight")
        w = np.broadcast_to(np.asarray(w, dtype=float), (n_test,))
        w0 = float(w[0]) if n_test else 0.0
        uniform = bool(n_test) and bool(np.allclose(w, w0, rtol=1e-9, atol=0))
        n_bins = y / w0 if w0 else 0.0
        exp = w0 * np.sqrt(n_bins)
        rows.append(dict(sample=name, group=("signal" if name == signal else group.get(name, "-")), n_test=n_test,
                         w=w0, n_bins=n_bins, y=y, stat=float(np.sqrt(var)), exp=exp, naive=float(np.sqrt(n_bins)),
                         pois=float(np.sqrt(y)), check=(np.sqrt(var) / exp if exp > 0 else float("nan")), uniform=uniform))
    bkg = [r for r in rows[1:]]
    rows.append(dict(sample="Total bkg", group="", n_test=sum(r["n_test"] for r in bkg), w=float("nan"),
                     n_bins=sum(r["n_bins"] for r in bkg), y=sum(r["y"] for r in bkg),
                     stat=float(np.sqrt(sum(r["stat"] ** 2 for r in bkg))), exp=float("nan"), naive=float("nan"),
                     pois=float(np.sqrt(sum(r["y"] for r in bkg))), check=float("nan"), uniform=True))
    return rows


def print_sample_table(rows):
    print("\n  Without binning (summed over all NN bins): stat must be w*sqrt(N_bins), NOT sqrt(N_bins)")
    print(f"  {'sample':<12}{'group':<9}{'N_test':>8}{'N_bins':>9}{'w/event':>11}{'yield':>12}{'stat (hist)':>13}"
          f"{'w*sqrt(N)':>12}{'sqrt(N)':>10}{'sqrt(yield)':>12}{'check':>8}")
    for r in rows:
        chk = "-" if not np.isfinite(r["check"]) else (f"{r['check']:.3f}" if r["uniform"] else f"{r['check']:.3f}*")
        print(f"  {r['sample']:<12}{r['group']:<9}{r['n_test']:>8d}{r['n_bins']:>9.0f}{r['w']:>11.4g}{r['y']:>12.5g}"
              f"{r['stat']:>13.4g}{r['exp']:>12.4g}{r['naive']:>10.4g}{r['pois']:>12.4g}{chk:>8}")
    bad = [r["sample"] for r in rows if np.isfinite(r["check"]) and r["uniform"] and abs(r["check"] - 1) > 1e-6]
    odd = [r["sample"] for r in rows if not r["uniform"]]
    print("  -> stat = w*sqrt(N) confirmed for every equal-weight sample" if not bad else f"  -> MISMATCH (check != 1) for: {', '.join(bad)}")
    if odd:
        print(f"  (* non-uniform weights, w*sqrt(N) not exact: {', '.join(odd)})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=[500])
    p.add_argument("--groups", action="store_true", help="use the config.BACKGROUNDS groups instead of single processes")
    a = p.parse_args()

    dataset, wm = PrepareFeaturesForTraning(), WeightManager()
    out_dir = RESULTS_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for m in a.masses:
        signal = f"Signal_{m}"
        # same two steps as tttg/limits/pyhf_limits.py
        scaled = dataset.get_test_with_sample(signal=signal, cat=a.cat)
        hists = get_histograms(m, scaled, a.cat)
        procs = {s: [s] for lst in BACKGROUNDS.values() for s in lst}
        mc = BACKGROUNDS if a.groups else procs
        hd = ExtractHistData(hists).extract_hist_info(MC_BCs=mc, signals=[signal], syst=False)
        hd_proc = hd if not a.groups else ExtractHistData(hists).extract_hist_info(MC_BCs=procs, signals=[signal], syst=False)

        cols = {signal: (hd.signals[signal], np.sqrt(hd.signals_var[signal]))}
        for name in mc:
            cols[name] = (hd.backgrounds[name], np.sqrt(hd.backgrounds_var[name]))
        cols["Total bkg"] = (np.asarray(hd.total_backgrounds), np.sqrt(hd.bkg_var))

        names = list(cols)
        edges = hd.bins
        print(f"\n{a.cat} {signal}:  yield ± stat per NN bin")
        print("  bin             " + "".join(f"{n:>24}" for n in names))
        for i in range(len(edges) - 1):
            print(f"  [{edges[i]:.3f},{edges[i + 1]:.3f}) " + "".join(
                f"{cols[n][0][i]:>14.4g} ± {cols[n][1][i]:<7.2g}" for n in names))

        path = out_dir / f"yield_stat_{a.cat}_{signal}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["lo", "hi"] + [f"{n} {k}" for n in names for k in ("yield", "stat")])
            for i in range(len(edges) - 1):
                w.writerow([edges[i], edges[i + 1]] + [x for n in names for x in (cols[n][0][i], cols[n][1][i])])
        print(f"saved {path}")

        rows = sample_table(hd_proc, scaled, wm, a.cat, signal, procs)
        print_sample_table(rows)
        path = out_dir / f"yield_stat_total_{a.cat}_{signal}.csv"
        keys = ["sample", "group", "n_test", "n_bins", "w", "y", "stat", "exp", "naive", "pois", "check"]
        with open(path, "w", newline="") as f:
            w_ = csv.writer(f)
            w_.writerow(["sample", "group", "N_test", "N_in_bins", "w_per_event", "yield", "stat_hist", "w_sqrtN",
                         "sqrtN_naive", "sqrt_yield", "stat_over_w_sqrtN"])
            w_.writerows([[r[k] for k in keys] for r in rows])
        print(f"saved {path}")


if __name__ == "__main__":
    main()