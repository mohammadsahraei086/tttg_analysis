"""Table of the NN output: nominal yield +- statistical uncertainty of the signal and of EACH background process, per NN bin.

Run from the diagnostics folder ...

    cd diagnostics
    python nn_yield_table.py --cat noEFT --mass 500

... or from the project root:   python -m diagnostics.nn_yield_table --cat noEFT --mass 500

It uses exactly the histograms that are given to the limit (same code as diagnostics/bkg_nn_histograms.py, which mirrors
tttg/histograms/nn_histograms.py): TEST half of every sample, mean sigmoid output of the seed ensemble, nominal weights
xsec * lumi * 2 / N_gen, config.NN_BINS bins on [0, 1).

  yield        = sum of the event weights in the bin                    (events at L = config.LUMINOSITY)
  stat. uncert.= sqrt(sum of the squared event weights in the bin)      (the MC statistical uncertainty that enters pyhf)
  "Total bkg"  = sum of the processes in config.BACKGROUNDS (what the limit uses), uncertainties added in quadrature

Two binnings are written: the NN bins as filled and, if they differ, the bins pyhf finally uses after rebin().
The signal is normalised to the cross section stored in the coffea metadata (i.e. mu = 1).

Output (results/diagnostics/):
  nn_yield_table_<cat>_Signal_<m>.md          readable tables (yield +- stat, relative uncertainty, raw MC events)
  nn_yield_table_<cat>_Signal_<m>_<binning>.csv   wide table, one row per bin, cells "yield +- unc"
  nn_yield_table_<cat>_Signal_<m>_long.csv    tidy table for Excel / pandas: one row per (binning, bin, process) with
                                              yield, stat_unc, rel_unc, n_mc_events as numbers
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tttg.config import CATS, LUMINOSITY, MASSES, NN_BINS, RESULTS_DIR
from diagnostics.bkg_nn_histograms import analyse, ensemble_scores, group_of, nominal_weights

OUT_DIR = RESULTS_DIR / "diagnostics"


# ----------------------------------------------------------------------------- pure helpers
def merge_groups_from_edges(edges, merged):
    """List of index arrays: which original bins make up each merged bin."""
    edges, merged = np.asarray(edges), np.asarray(merged)
    return [np.where((edges[:-1] >= lo - 1e-9) & (edges[:-1] < hi - 1e-9))[0] for lo, hi in zip(merged[:-1], merged[1:])]


def collect(res, groups):
    """{column: (yield, stat_unc, n_mc)} per output bin, columns: signal, every background sample (by yield), Total bkg."""
    def agg(d):
        y = np.array([d["yield"][g].sum() for g in groups])
        v = np.array([d["sumw2"][g].sum() for g in groups])
        n = np.array([d["nraw"][g].sum() for g in groups])
        return y, np.sqrt(v), n

    order = sorted(res["samples"], key=lambda s: -res["samples"][s]["yield_all"])
    cols = {res["signal"]: agg(res["signal_hist"])}
    for s in order:
        cols[s] = agg(res["samples"][s])
    used = res["in_limit"]
    if used:
        cols["Total bkg"] = (sum(cols[s][0] for s in used),
                             np.sqrt(sum(cols[s][1] ** 2 for s in used)),
                             sum(cols[s][2] for s in used))
    return cols


def pm(y, e, n):
    """'yield +- unc'; a bin without any MC event is just '0'."""
    return "0" if n == 0 else f"{y:.4g} ± {e:.2g}"


def rel(y, e, n):
    return "-" if n == 0 or y <= 0 else f"{100 * e / y:.1f}%"


def bin_labels(edges, groups):
    return [f"[{edges[g[0]]:.3f}, {edges[g[-1] + 1]:.3f})" for g in groups]


def md_table(labels, cols, cell):
    names = list(cols)
    rows = ["| NN bin | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for i, lab in enumerate(labels):
        rows.append(f"| {lab} | " + " | ".join(cell(*(a[i] for a in cols[c])) for c in names) + " |")
    return "\n".join(rows)


def write_outputs(res, cat, mass, merged_ok, out_dir=OUT_DIR):
    edges = res["edges"]
    raw_groups = [np.array([i]) for i in range(len(edges) - 1)]
    binnings = {"nn_bins": raw_groups}
    if merged_ok:
        mg = merge_groups_from_edges(edges, res["merged_edges"])
        if len(mg) != len(raw_groups):
            binnings["pyhf_bins"] = mg
    stem = f"nn_yield_table_{cat}_{res['signal']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    md = [f"# NN-output yields: {cat}, {res['signal']}",
          "",
          f"* yield = sum of the event weights in the bin (nominal, L = {LUMINOSITY} fb^-1, TEST half of the samples, "
          "mean of the seed-ensemble NN output); uncertainty = sqrt(sum of squared weights) = MC statistical uncertainty",
          "* signal normalised to the cross section in the coffea metadata (mu = 1)",
          f"* Total bkg = {', '.join(res['in_limit'])} (config.BACKGROUNDS; uncertainties added in quadrature)",
          ]
    if res["not_in_limit"]:
        md.append(f"* NOT in Total bkg (not in config.BACKGROUNDS): {', '.join(res['not_in_limit'])}")
    lost = [f"{s}: {d['n_over'] + d['n_nan'] + d['n_under']}" for s, d in
            [(res["signal"], res["signal_hist"])] + list(res["samples"].items()) if d["n_over"] + d["n_nan"] + d["n_under"]]
    if lost:
        md.append(f"* events with NN output >= 1.0, < 0 or NaN are in no bin (as in the limit histograms): {'; '.join(lost)}")
    md.append("* a cell '0' means no MC event in that bin (a yield of 0 does NOT mean a vanishing uncertainty of the prediction)")

    long_rows = []
    for name, groups in binnings.items():
        cols = collect(res, groups)
        labels = bin_labels(edges, groups)
        title = "NN bins as filled" if name == "nn_bins" else "bins used by pyhf after rebin()"
        md += ["", f"## {title}: yield ± statistical uncertainty", "", md_table(labels, cols, pm),
               "", f"## {title}: relative statistical uncertainty", "", md_table(labels, cols, lambda y, e, n: rel(y, e, n)),
               "", f"## {title}: number of MC events", "", md_table(labels, cols, lambda y, e, n: str(int(n)))]
        with open(out_dir / f"{stem}_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["NN bin"] + list(cols))
            for i, lab in enumerate(labels):
                w.writerow([lab] + [pm(*(a[i] for a in cols[c])) for c in cols])
        for i, g in enumerate(groups):
            for c, (y, e, n) in cols.items():
                grp = "signal" if c == res["signal"] else ("total background" if c == "Total bkg" else
                                                            (group_of(c) or "not in limit"))
                long_rows.append([name, i, edges[g[0]], edges[g[-1] + 1], c, grp, y[i], e[i],
                                  (e[i] / y[i] if y[i] > 0 else ""), int(n[i])])
    with open(out_dir / f"{stem}_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["binning", "bin", "lo", "hi", "process", "group", "yield", "stat_unc", "rel_unc", "n_mc_events"])
        w.writerows(long_rows)
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n")
    return stem, list(binnings), md


def print_summary(res, cols, labels):
    sig = res["signal"]
    print(f"\n  {'NN bin':<16}{sig + ' (±stat)':>26}{'Total bkg (±stat)':>26}{'bkg rel.':>10}{'bkg N_MC':>10}")
    for i, lab in enumerate(labels):
        ys, es, ns = (a[i] for a in cols[sig])
        yb, eb, nb = (a[i] for a in cols["Total bkg"])
        print(f"  {lab:<16}{pm(ys, es, ns):>26}{pm(yb, eb, nb):>26}{rel(yb, eb, nb):>10}{int(nb):>10d}")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--masses", "--mass", nargs="+", type=int, choices=MASSES, default=[500], dest="masses",
                   help="signal mass(es) (default 500)")
    p.add_argument("--bins", type=int, default=NN_BINS, help=f"number of NN bins (default config.NN_BINS = {NN_BINS})")
    a = p.parse_args()

    from tttg.data.features import PrepareFeaturesForTraning
    from tttg.data.weights import WeightManager

    ds, wm = PrepareFeaturesForTraning(), WeightManager()
    for m in a.masses:
        signal = f"Signal_{m}"
        scaled = ds.get_test_with_sample(signal=signal, cat=a.cat)
        scores, paths = ensemble_scores(a.cat, m, scaled)
        weights = nominal_weights(wm, a.cat, scores)
        print(f"\n{a.cat} {signal}: ensemble of {len(paths)} model(s)")
        res = analyse(scores, weights, signal, np.linspace(0.0, 1.0, a.bins + 1))
        stem, binnings, _ = write_outputs(res, a.cat, m, merged_ok=res["merged_edges"] is not None)
        for name in binnings:
            groups = ([np.array([i]) for i in range(a.bins)] if name == "nn_bins"
                      else merge_groups_from_edges(res["edges"], res["merged_edges"]))
            print(f"\n  {'NN bins as filled' if name == 'nn_bins' else 'pyhf bins after rebin()'}")
            print_summary(res, collect(res, groups), bin_labels(res["edges"], groups))
        print(f"\nsaved results/diagnostics/{stem}.md, {stem}_{{{','.join(binnings)}}}.csv and {stem}_long.csv")


if __name__ == "__main__":
    main()
