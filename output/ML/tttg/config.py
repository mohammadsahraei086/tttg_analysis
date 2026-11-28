"""Shared configuration for the pp -> tttg limit chain (single-lepton channel).

Every path is anchored to the project root (the folder that contains ``tttg/``), so the code
works from any working directory.
"""
import os
import re
from pathlib import Path

# ---- project layout --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"            # <cat>/Signal_<mass>.pth        (training/)
ROOT_OUT_DIR = PROJECT_ROOT / "output"         # <cat>/mT<mass>/output_root/    (limits/root_files.py)
RESULTS_DIR = PROJECT_ROOT / "results"         # limit json files
INPUTS_DIR = PROJECT_ROOT / "inputs"           # external inputs (cross-section scan)
PLOT_DIR = PROJECT_ROOT / "plots"
TRAIN_PLOT_DIR = PLOT_DIR / "training"         # <cat>/training_curves_<m>.png, nn_performance_<m>.png
NN_HIST_PLOT_DIR = PLOT_DIR / "nn_histograms"  # <cat>/Signal_<m>_normalized.png
LIMIT_PLOT_DIR = PLOT_DIR / "limits"           # xsec_limit_*.png, g_limit_*.png, compare_*.png

LIMITS_JSON = RESULTS_DIR / "limits_pyhf.json"             # written by limits/pyhf_limits.py
LIMITS_COMBINE_JSON = RESULTS_DIR / "limits_combine.json"  # written by limits/combine.py
G_LIMITS_JSON = RESULTS_DIR / "g_limits.json"              # written by limits/coupling.py
XSEC_TABLE = INPUTS_DIR / "tttg_CrossSection.txt"          # sigma(g/Lambda) scan from MadGraph

# The coffea file with features, weights and cross sections lives one level above the project
# (as before). Set TTTG_COFFEA to point somewhere else.
COFFEA_OUTPUT = Path(os.environ.get("TTTG_COFFEA", PROJECT_ROOT.parent / "output.coffea"))

# ---- analysis grid ---------------------------------------------------------
MASSES = [500, 800, 1000, 1200, 1500, 2000, 2500, 3000]
CATS = ["noEFT", "sEFT", "s_hatEFT"]

CAT_LABELS = {
    "noEFT": "No EFT",
    "sEFT": r"$S_T<\Lambda$",
    "s_hatEFT": r"$\hat{s}<\Lambda$",
}

LUMINOSITY = 3000   # fb^-1 (phase 2)

BACKGROUNDS_old = {
    # "ttt": ["tttJets"],
    "tt": ["tt"],
    # "tttt": ["tttt"],
    # "ttV": ["ttZ", "ttW"],
    # "VV+VVV": ["WZ", "WW", "ZZ", "WWW"],
    # "tWZ": ["tWZ"],
    # "ttH": ["ttH"],
    # "other": ["WJets", "tZJets", "tHJets", "HW"], # , "DYJets"
}

BACKGROUNDS = {
    r"$t\bar{t}$": ["tt_loose_cut", 'tt_tight_cut', "tt_2lep"],
    r"$VV+VVV+VH$": ["WZ", "WW", "ZZ", "WWW", "HW"],
    r"$t\bar{t}H$": ["ttH"],
    r"$t\bar{t}Z$": ["ttZ"],
    r"$t\bar{t}W$": ["ttW"],
    r"$t\bar{t}t\bar{t}$": ["tttt"],
    r"$tWZ$": ["tWZ"],
    r"$tXj~(X = Z,H)$": ["tZJets", "tHJets"],
    r"$tttj$": ["tttJets"],
    r"V+jets":  ["WJets", "DYJets"] # 
}

# Relative normalisation uncertainty on the backgrounds in the pyhf limits. Default for every
# group; override single groups in BKG_REL_UNC_BY_GROUP, e.g. {"tt": 0.2, "other": 0.5}.
# (Only relevant when the groups are treated separately: with --bkg-grouping total the single
# summed background always uses BKG_REL_UNC.)
BKG_REL_UNC = 0.3
BKG_REL_UNC_BY_GROUP: dict[str, float] = {}

# Background grouping used only for the NN-output control plots (plots/nn_histograms)
NN_PLOT_BACKGROUNDS = {
    "ttt": ["tttJets"],
    "tt": ["tt"],
    "tttt": ["tttt"],
    r"ttV+tVJets": ["ttZ", "ttW", "tZJets"],
    "VV+VVV": ["WZ", "WW", "ZZ", "WWW"],
    "tWZ": ["tWZ"],
    "ttH+tHJets": ["ttH", "tHJets"],
    "HW": ["HW"],
    "VJets": ["WJets", "DYJets"],
}

# background category -> file stem; the histogram inside is named nn_score_<stem>
# (these are also the process names in the Combine datacard)
BKG_FILE_NAMES = {
    "ttt": "bg_tttj_1lepton",
    "tt": "bg_tt_1lepton",
    "tttt": "bg_tttt",
    "ttV": "bg_ttV_1lepton",
    "VV+VVV": "bg_VV_VVV_1lepton",
    "tWZ": "bg_tWZ",
    "ttH": "bg_tth",
    "other": "other",
}

# key in hd.syst[signal]  ->  name used in the signal histograms (nn_score_signal__<name>Up/Down)
SYST_NAMES = {
    "mur_unc": "muR",
    "muf_unc": "muF",
    "isr_unc": "isr",
    "fsr_unc": "fsr",
    "pdf_unc": "pdf",
}

# signal theory systematics (keys of hd.syst[signal] are f"{name}_unc")
SIG_SYSTS = ["pdf", "mur", "muf", "isr", "fsr"]

# order of the expected-limit list stored in the limit json files
BAND_INDEX = {"minus2": 0, "minus1": 1, "median": 2, "plus1": 3, "plus2": 4}

# ---- NN histogram binning --------------------------------------------------
NN_BINS = 15

# ---- training --------------------------------------------------------------
EPOCHS = 40
PATIENCE = 15
RF_BENCHMARK_AUC = 0.8359   # dashed reference line in the training-curve plots

# How the physical (cross-section) weights enter the training. Both can be overridden on the command
# line of tttg.training.train_eval (--scheme, --min-events / --keep-low-stat, --seed). See CHANGES.md.
#   "baseline"  : every background event counts the same (the old behaviour, classes still balanced)
#   "proc:<a>"  : every background PROCESS gets a total weight ~ (its yield)^a, spread uniformly over its
#                 events.  a=1 -> physical composition, a=0.5 -> tempered (default), a=0 -> all processes equal
TRAIN_SCHEME = "proc:0.5"
# Background processes with fewer TRAINING events than this are left out of the training (they are still
# evaluated and still enter the limit histograms). 0 = keep every process.
TRAIN_MIN_EVENTS = 100
TRAIN_SEED = 0              # torch / numpy seed (initial weights, batch order, dropout)
# Seed ensemble: N_SEEDS networks per (cat, mass) are trained with seeds TRAIN_SEED, TRAIN_SEED+1, ... and their
# sigmoid outputs are averaged (tttg.training.train_eval.test and tttg.histograms.nn_histograms).
N_SEEDS = 1


N_FOLDS = 2   # 2-fold cross-training: fold f = networks trained on the OTHER half, applied to half f (see CHANGES.md)


def model_path(cat: str, mass, seed=None, fold=None) -> Path:
    """models/<cat>/Signal_<mass>.pth (a single model), with a seed one ensemble member
    models/<cat>/Signal_<mass>_seed<seed>.pth, and with a fold the member of the 2-fold training
    models/<cat>/Signal_<mass>_fold<fold>[_seed<seed>].pth"""
    suffix = "" if fold is None else f"_fold{fold}"
    suffix += "" if seed is None else f"_seed{seed}"
    return MODEL_DIR / cat / f"Signal_{mass}{suffix}.pth"


def model_paths(cat: str, mass, fold=None) -> list:
    """Files of the NN ensemble of (cat, mass), ordered by seed.

    fold=None (old behaviour): Signal_<mass>_seed<k>.pth if they exist, otherwise the single model Signal_<mass>.pth.
    fold=0/1 : only the members of that fold, Signal_<mass>_fold<fold>_seed<k>.pth (no fallback: a missing fold
               model must be an error, never silently another fold's network). Empty list: no model.
    """
    tag = "" if fold is None else f"_fold{fold}"
    pattern = re.compile(rf"Signal_{mass}{tag}_seed(\d+)\.pth$")
    members = []
    for p in (MODEL_DIR / cat).glob(f"Signal_{mass}{tag}_seed*.pth"):
        hit = pattern.search(p.name)
        if hit:
            members.append((int(hit.group(1)), p))
    if members:
        return [p for _, p in sorted(members)]
    if fold is not None:
        return []
    single = model_path(cat, mass)
    return [single] if single.exists() else []


# ---- cross-section table ---------------------------------------------------
XSEC_MASS_COLUMN = "mass#9000017"
XSEC_G_COLUMN = "frblock#3"
XSEC_VALUE_COLUMN = "cross"
XSEC_UNIT_FACTOR = 1e3        # table is in pb, limits are in fb
LAMBDA_TEV = 5.0              # Lambda = 5000 GeV, so g/Lambda = g / 5 [TeV^-1]

# ---- Combine cross-check ---------------------------------------------------
DATACARD_NAME = "tttg_limit.txt"
DATACARD_BIN = "SR_1lepton"
BKG_LNN = 1.3                 # flat normalisation uncertainty on every background (lnN)
AUTOMCSTATS = 10              # threshold of "* autoMCStats"
# extra arguments appended to the combine command. The pyhf limits (mu ~ 10-80) are above
# Combine's default POI range, so the range is widened. Set to [] for the plain command.
COMBINE_ARGS = ["--rMax", "200"]