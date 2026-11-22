"""
Write ROOT files (single-lepton channel) from the NN-output histograms.

For every category and signal mass:
    <ROOT_OUT_DIR>/<cat>/mT<mass>/output_root/
        bg_<...>.root          one file per background category (nominal only)
        signal.root            nominal + Up/Down for muR, muF, isr, fsr, pdf
        data_obs.root          pseudo-data = sum of backgrounds (Combine wants a data_obs histogram)
        NN_output.root         hadd of signal + all backgrounds + data_obs
        total_background.root  sum of all backgrounds in one histogram

All histograms carry proper bin errors (sqrt of the sum of squared weights). Combine's
"autoMCStats" reads them; without them ROOT would fall back to sqrt(bin content), which is
wrong for weighted events.
"""
import os
import shutil
import subprocess
from array import array

import numpy as np
import ROOT

from tttg.config import BACKGROUNDS, BKG_FILE_NAMES, CATS, MASSES, ROOT_OUT_DIR, SYST_NAMES
from tttg.data.features import PrepareFeaturesForTraning
from tttg.histograms.extractor import ExtractHistData
from tttg.histograms.nn_histograms import get_histograms
from tttg.limits.pyhf_limits import bin_merger

ROOT.gROOT.SetBatch(True)

BASE_DIR = os.path.abspath(ROOT_OUT_DIR)


def make_th1(name, values, edges, errors=None):
    edges_arr = array("d", [float(e) for e in edges])
    h = ROOT.TH1D(name, name, len(edges_arr) - 1, edges_arr)
    if errors is not None:
        h.Sumw2()
    for i, v in enumerate(values, start=1):
        h.SetBinContent(i, float(v))
        if errors is not None:
            h.SetBinError(i, float(errors[i - 1]))
    return h


def write_file(path, hists):
    f = ROOT.TFile(path, "RECREATE")
    for h in hists:
        h.SetDirectory(f)
        h.Write()
    f.Close()


def hadd_files(out_path, in_paths):
    if shutil.which("hadd"):
        subprocess.run(["hadd", "-f", out_path] + in_paths, check=True,
                       stdout=subprocess.DEVNULL)
        return
    # fallback if hadd is not on PATH: same-named objects are summed, others copied
    merged = {}
    for p in in_paths:
        f = ROOT.TFile(os.path.abspath(p))  # plain constructor: no remote-URL plugin lookup
        if f.IsZombie():
            raise RuntimeError(f"cannot open {p}")
        for key in f.GetListOfKeys():
            obj = key.ReadObj()
            if key.GetName() in merged:
                merged[key.GetName()].Add(obj)
            else:
                c = obj.Clone()
                c.SetDirectory(0)
                merged[key.GetName()] = c
        f.Close()
    write_file(out_path, list(merged.values()))


def sumw2(hists, samples):
    """Sum of squared weights per bin for the nominal histograms of `samples`."""
    return sum(hists[s][{"variation": "Nominal"}].variances() for s in samples)


def write_channel_files(hd, hists, signal_name, out_dir, rebin_bins=False):
    os.makedirs(out_dir, exist_ok=True)
    edges = np.asarray(hd.bins)
    if rebin_bins:   # merge low-statistics bins exactly as pyhf_limits.py does
        edges, merge = bin_merger(edges, hd.signals[signal_name], hd.total_backgrounds,
                                  hd.bkg_var, hd.syst[signal_name])
    else:
        merge = np.asarray
    written = []

    # ---- backgrounds: nominal only, with MC-stat errors
    for bkg, vals in hd.backgrounds.items():
        stem = BKG_FILE_NAMES[bkg]
        path = os.path.join(out_dir, f"{stem}.root")
        err = np.sqrt(merge(sumw2(hists, BACKGROUNDS[bkg])))
        write_file(path, [make_th1(f"nn_score_{stem}", merge(vals), edges, errors=err)])
        written.append(path)

    # ---- signal: nominal (with MC-stat errors) + Up/Down per systematic
    nominal = np.asarray(hd.signals[signal_name])
    sig_err = np.sqrt(merge(sumw2(hists, [signal_name])))
    syst = hd.syst[signal_name]
    sig_hists = [make_th1("nn_score_signal", merge(nominal), edges, errors=sig_err)]
    for key, label in SYST_NAMES.items():
        rel = np.asarray(syst[key])  # symmetrised relative uncertainty per bin
        up = nominal * (1.0 + rel)
        down = np.clip(nominal * (1.0 - rel), 0.0, None)
        sig_hists.append(make_th1(f"nn_score_signal__{label}Up", merge(up), edges))
        sig_hists.append(make_th1(f"nn_score_signal__{label}Down", merge(down), edges))
    sig_path = os.path.join(out_dir, "signal.root")
    write_file(sig_path, sig_hists)

    # ---- total background (and pseudo-data for Combine)
    total = np.sum([np.asarray(v) for v in hd.backgrounds.values()], axis=0)
    total_err = np.sqrt(merge(hd.bkg_var))
    total = merge(total)
    data_path = os.path.join(out_dir, "data_obs.root")
    write_file(data_path, [make_th1("data_obs", total, edges)])

    # ---- NN_output.root = hadd(signal + all backgrounds + data_obs)
    hadd_files(os.path.join(out_dir, "NN_output.root"), [sig_path] + written + [data_path])

    write_file(
        os.path.join(out_dir, "total_background.root"),
        [make_th1("nn_score_total_background", total, edges, errors=total_err)],
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Write the ROOT files / Combine inputs.")
    ap.add_argument("--rebin", action=argparse.BooleanOptionalAction, default=True,
                help="merge low-statistics NN bins (default: on; use --no-rebin for the raw 20 bins)")
    ap.add_argument("--out", default=str(ROOT_OUT_DIR),
                    help="output folder (default: config.ROOT_OUT_DIR); use e.g. output_rebin with --rebin")
    args = ap.parse_args()
    BASE_DIR = os.path.abspath(args.out)

    dataset = PrepareFeaturesForTraning()

    for cat in CATS:
        for m in MASSES:
            name = f"Signal_{m}"
            hists = get_histograms(m, dataset.get_test_with_sample(signal=name, cat=cat), cat)
            hd = ExtractHistData(hists).extract_hist_info(MC_BCs=BACKGROUNDS, signals=[name])

            out_dir = os.path.join(BASE_DIR, cat, f"mT{m}", "output_root")
            write_channel_files(hd, hists, name, out_dir, rebin_bins=args.rebin)
            print(f"[{cat}] mT{m}: wrote {out_dir}")
