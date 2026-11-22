# pp → tttg (single-lepton) — ML + limit chain

NN signal-vs-background classifiers (one per signal mass and EFT category) → NN-output histograms →
95 % CL limits with pyhf (cross-checked with Combine) → cross-section and g/Λ limits.

Everything is run **from the project root**, as modules (`python -m ...`). All paths are defined in
`tttg/config.py` and anchored to the project root, so the working directory does not matter.
The coffea file is expected at `../output.coffea` (next to this folder); set `TTTG_COFFEA=/path/to/output.coffea` to change it.

## Layout

```
ML/
├── tttg/                       the code (a python package)
│   ├── config.py               masses, categories, backgrounds, systematics, ALL paths
│   ├── data/
│   │   ├── features.py         PrepareFeaturesForTraning: coffea -> train/val/test arrays
│   │   └── weights.py          WeightManager (xsec*lumi/Ngen, scale/PDF/ISR/FSR weights)
│   ├── training/
│   │   ├── model.py            OptimizedHEPClassifier (MLP)
│   │   ├── trainer.py          OptimizedTrainer (AdamW + OneCycle, early stopping)
│   │   └── train_eval.py       train + test every (cat, mass); training/ROC plots
│   ├── histograms/
│   │   ├── nn_histograms.py    get_histograms(): NN output -> weighted hist (+ variations)
│   │   ├── systematics.py      muR / muF / PDF / ISR / FSR uncertainties
│   │   ├── extractor.py        HistInfo, ExtractHistData
│   │   └── plotter.py          NN-output control plots
│   ├── limits/
│   │   ├── pyhf_limits.py      rebin + pyhf model + CLs scan   -> results/limits_pyhf.json
│   │   ├── root_files.py       ROOT files / Combine inputs      -> output/
│   │   ├── combine.py          datacards + Combine AsymptoticLimits -> results/limits_combine.json
│   │   ├── coupling.py         sigma(g) fit, g/Λ limits         -> results/g_limits.json
│   │   └── io.py               load limit json / signal xsecs, mu limit -> xsec limit
│   └── plotting/
│       ├── style.py            colours, axis style
│       ├── xsec_limits.py      cross-section limit plots (bands + theory)
│       ├── g_limits.py         g/Λ limit plots
│       └── compare.py          pyhf vs Combine
├── inputs/tttg_CrossSection.txt   σ(g/Λ) scan from MadGraph
├── models/<cat>/Signal_<m>.pth    trained networks
├── output/<cat>/mT<m>/output_root/   ROOT files + datacards (Combine inputs)
├── results/                    limits_pyhf.json, limits_combine.json, g_limits.json
├── plots/
│   ├── training/<cat>/         training_curves_<m>.png, nn_performance_<m>.png
│   ├── nn_histograms/<cat>/    Signal_<m>_normalized.png
│   └── limits/                 xsec_limit_*.png, g_limit_*.png, compare_pyhf_combine.png
├── notebooks/legacy_train_test.ipynb   old scratch notebook (superseded)
└── archive/                    old backups (root-file zip, old limits json)
```

## Pipeline

| # | step | command | reads | writes |
|---|------|---------|-------|--------|
| 1 | train + test NNs (cross-section weighted, see `CHANGES.md`) | `python -m tttg.training.train_eval [--cats ..] [--masses ..] [--scheme proc:0.5\|baseline\|proc:<a>] [--min-events N \| --keep-low-stat] [--seed S]` | coffea | `models/` (`.pth` + `.json` with the settings), `plots/training/` |
| 2 | NN-output control plots | `python -m tttg.histograms.plotter` | coffea, `models/` | `plots/nn_histograms/` |
| 3 | pyhf limits on μ | `python -m tttg.limits.pyhf_limits [--bkg-mode norm\|uncorrelated] [--bkg-grouping total\|groups] [--no-sig-stat]` | coffea, `models/` | `results/limits_pyhf.json` |
| 4 | ROOT files / Combine inputs | `python -m tttg.limits.root_files [--no-rebin] [--out DIR]` | coffea, `models/` | `output/` |
| 5 | Combine cross-check | `python -m tttg.limits.combine [--cats ..] [--masses ..] [--cards-only]` | `output/` | `results/limits_combine.json` |
| 6 | cross-section limit plots | `python -m tttg.plotting.xsec_limits [--cats ..] [--separate] [--no-bands] [--limits FILE]` | `results/`, coffea | `plots/limits/` |
| 7 | g/Λ limits | `python -m tttg.limits.coupling [--show]` | `results/limits_pyhf.json`, `inputs/`, coffea | `results/g_limits.json`, `plots/limits/` |
| 8 | pyhf vs Combine | `python -m tttg.plotting.compare` | both limit jsons | `plots/limits/compare_pyhf_combine.png` |

**pyhf options (step 3)**
- `--bkg-grouping total` (default): all backgrounds summed into one pyhf sample. `groups`: one sample per group in `config.BACKGROUNDS` (output goes to `results/limits_pyhf_groups.json` unless `--out` is given).
- --bkg-mode norm (default): one normalisation nuisance shared by all backgrounds (fully correlated, `config.BKG_REL_UNC` = 30 %, per-group overrides in `BKG_REL_UNC_BY_GROUP`) + MC-stat per sample. `uncorrelated`: the 30 % is applied per bin and per sample, uncorrelated.
- Signal MC-statistical uncertainty is included by default (`--no-sig-stat` to switch it off).

Steps 3 and 4 share the same rebinning (`bin_merger` in `pyhf_limits.py`), so the Combine inputs match the pyhf model.

## Using it from a notebook (e.g. SWAN)

```python
import sys; sys.path.insert(0, "/path/to/ML")
from tttg.config import MASSES, CATS
from tttg.limits.io import load_limits
```

## Requirements

See `requirements.txt`. PyROOT is only needed for step 4, `combine` on `PATH` only for step 5.

## Diagnostics (read-only checks, not part of the analysis)

`diagnostics/` only *reads* from `tttg/` (models, data, weights) and never changes it. Run from the project root:

| check | command |
|---|---|
| weighted vs unweighted performance, AUC per background, cut scan | `python -m diagnostics.weighted_performance --cat noEFT --masses 500 800` |
| what produces spikes in the NN output | `python -m diagnostics.score_spikes --cat noEFT --mass 800` |

Plots go to `plots/diagnostics/`.
