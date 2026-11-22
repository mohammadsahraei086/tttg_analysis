"""Cross-check of the pyhf limits with the Combine tool.

For every <ROOT_OUT_DIR>/<cat>/mT<mass>/output_root/ written by limits/root_files.py this script
  1. writes the datacard  tttg_limit.txt  next to NN_output.root
  2. runs   combine -M AsymptoticLimits tttg_limit.txt -m <mass> --run blind -n _VLT_M<mass>
  3. reads the limits back from the higgsCombine*.root file Combine produces
and saves everything in results/limits_combine.json, in the same format as limits_pyhf.json
("expected" = [-2 sigma, -1 sigma, median, +1 sigma, +2 sigma] limit on the signal strength mu).

`combine` must be on your PATH (see the Combine docs for how to set it up in each new shell).

Examples:
    python -m tttg.limits.combine                              # all categories and masses
    python -m tttg.limits.combine --cats noEFT --masses 500 1000
    python -m tttg.limits.combine --cards-only                 # only write the datacards, run nothing
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tttg.config import (AUTOMCSTATS, BKG_FILE_NAMES, BKG_LNN, CATS, COMBINE_ARGS, DATACARD_BIN,
                    DATACARD_NAME, LIMITS_COMBINE_JSON, MASSES, ROOT_OUT_DIR, SYST_NAMES)

SIGNAL = "nn_score_signal"
BACKGROUND_PROCS = [f"nn_score_{stem}" for stem in BKG_FILE_NAMES.values()]
QUANTILES = [0.025, 0.16, 0.5, 0.84, 0.975]    # -2s, -1s, median, +1s, +2s


# ---------------------------------------------------------------------------
# datacard
# ---------------------------------------------------------------------------
def make_datacard() -> str:
    """Datacard text. Process names are the histogram names inside NN_output.root."""
    procs = [SIGNAL] + BACKGROUND_PROCS
    n = len(procs)
    w = max(len(p) for p in procs) + 2

    def row(label, cells, kind=""):
        return f"{label:<10}{kind:<7}" + "".join(f"{c:<{w}}" for c in cells)

    nuis = [row("norm", ["-"] + [str(BKG_LNN)] * (n - 1), "lnN")]
    for label in SYST_NAMES.values():            # signal-only shape systematics
        # "_isr" -> Combine looks for  nn_score_signal__isrUp / nn_score_signal__isrDown
        nuis.append(row(f"_{label}", ["1"] + ["-"] * (n - 1), "shape"))

    lines = [
        "imax 1  number of categories",
        f"jmax {n - 1}  number of backgrounds",
        "kmax *  number of nuisance parameters",
        "-" * 40,
        "shapes * * NN_output.root $PROCESS $PROCESS_$SYSTEMATIC",
        "-" * 40,
        f"bin         {DATACARD_BIN}",
        "observation -1",
        "-" * 40,
        row("bin", [DATACARD_BIN] * n),
        row("process", procs),
        row("process", [str(i) for i in range(n)]),
        row("rate", ["-1"] * n),
        "-" * 40,
        *nuis,
        "",
        f"* autoMCStats {AUTOMCSTATS}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# reading ROOT files (uproot if installed, otherwise PyROOT)
# ---------------------------------------------------------------------------
def root_keys(path) -> set[str]:
    try:
        import uproot
        with uproot.open(path) as f:
            return {k.split(";")[0] for k in f.keys()}
    except ImportError:
        import ROOT
        f = ROOT.TFile.Open(str(path))
        names = {k.GetName() for k in f.GetListOfKeys()}
        f.Close()
        return names


def read_limit_tree(path) -> list[tuple[float, float]]:
    """[(quantileExpected, limit), ...] from the 'limit' tree of a higgsCombine file."""
    try:
        import uproot
        with uproot.open(path) as f:
            t = f["limit"].arrays(["quantileExpected", "limit"], library="np")
        return list(zip(t["quantileExpected"].tolist(), t["limit"].tolist()))
    except ImportError:
        import ROOT
        f = ROOT.TFile.Open(str(path))
        rows = [(e.quantileExpected, e.limit) for e in f.Get("limit")]
        f.Close()
        return rows


def check_inputs(workdir: Path) -> None:
    """Fail early if NN_output.root lacks a histogram the datacard asks for."""
    keys = root_keys(workdir / "NN_output.root")
    needed = [SIGNAL] + BACKGROUND_PROCS
    needed += [f"{SIGNAL}__{s}{d}" for s in SYST_NAMES.values() for d in ("Up", "Down")]
    missing = [h for h in needed if h not in keys]
    if missing:
        raise RuntimeError(f"NN_output.root is missing {missing}")
    if "data_obs" not in keys:
        print("   warning: no data_obs histogram in NN_output.root; if combine complains, "
              "re-run python -m tttg.limits.root_files (it now adds one)")


# ---------------------------------------------------------------------------
# running combine
# ---------------------------------------------------------------------------
def run_combine(workdir: Path, mass: int) -> Path:
    cmd = ["combine", "-M", "AsymptoticLimits", DATACARD_NAME, "-m", str(mass),
           "--run", "blind", "-n", f"_VLT_M{mass}", *COMBINE_ARGS]
    res = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    log = workdir / f"combine_M{mass}.log"
    log.write_text(" ".join(cmd) + "\n\n" + res.stdout + res.stderr)
    if res.returncode != 0:
        tail = "\n".join((res.stdout + res.stderr).strip().splitlines()[-8:])
        raise RuntimeError(f"combine failed (full output in {log}):\n{tail}")
    found = sorted(workdir.glob(f"higgsCombine_VLT_M{mass}.AsymptoticLimits.mH*.root"))
    if not found:
        raise RuntimeError(f"combine ran but no higgsCombine_VLT_M{mass}*.root appeared in {workdir}")
    return found[-1]


def expected_limits(higgs_file: Path) -> list[float]:
    by_quantile = {round(float(q), 3): float(lim) for q, lim in read_limit_tree(higgs_file)}
    return [by_quantile[q] for q in QUANTILES]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cats", nargs="+", choices=CATS, default=CATS)
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=MASSES)
    p.add_argument("--root-dir", type=Path, default=ROOT_OUT_DIR,
                   help="folder with <cat>/mT<mass>/output_root/ (default: config.ROOT_OUT_DIR)")
    p.add_argument("--out", type=Path, default=LIMITS_COMBINE_JSON)
    p.add_argument("--cards-only", action="store_true", help="write the datacards and stop")
    args = p.parse_args()

    if not args.cards_only and shutil.which("combine") is None:
        raise SystemExit("`combine` not found on PATH. Set up your Combine environment first "
                         "(or use --cards-only to just write the datacards).")

    results = {cat: {} for cat in args.cats}
    for cat in args.cats:
        for m in args.masses:
            workdir = args.root_dir / cat / f"mT{m}" / "output_root"
            print(f"[{cat}] mT{m}")
            if not (workdir / "NN_output.root").exists():
                print(f"   skipped: {workdir}/NN_output.root not found")
                continue
            try:
                check_inputs(workdir)
                (workdir / DATACARD_NAME).write_text(make_datacard())
                if args.cards_only:
                    print(f"   wrote {workdir / DATACARD_NAME}")
                    continue
                exp = expected_limits(run_combine(workdir, m))
                results[cat][str(m)] = {"expected": exp}
                print("   expected (-2s,-1s,median,+1s,+2s):", [round(x, 3) for x in exp])
            except Exception as e:
                print(f"   FAILED: {e}")

    if not args.cards_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
