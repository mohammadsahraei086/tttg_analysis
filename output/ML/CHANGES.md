# CHANGES.md – cross-section weights in the NN training

This file is the log of every change made to the main code (`tttg/`) after the diagnostics in
`diagnostics/` showed that the NN must see the physical (cross-section) background composition.
Nothing in `diagnostics/` was modified. The original code is untouched in your zip/backup; every change below
is marked in the code with a comment saying what it is for.

## 1. Why (the problem)

* The NN used to be trained on raw MC events: every background event counted the same, only the
  signal/background balance was corrected (`pos_weight`). The early-stopping AUC was unweighted too.
* The limit, however, fills its histograms with `xsec * lumi * 2 / N_gen` per event. In that
  composition tt is ~90 % of the background yield, while some processes have few events with huge weights.
* So the network optimised a background mixture that is not the one the limit sees
  (diagnostics: unweighted AUC 0.78 vs physically weighted 0.76 at m=800, much worse against tt).
* `diagnostics/weighted_training.py` + `experiment_limits.py` showed that giving each background *process*
  a total weight ~ (its yield)^0.5 (`proc:0.5`) and dropping processes with < 100 training events improves the
  weighted performance and, in most masses (clearly at 1000 and 1500), the pyhf limit. The seed-to-seed
  spread is large (e.g. m=500 was unstable), so compare limits over several seeds.
  This change moves exactly that recipe into the real training.

## 2. How to use it

```
python -m tttg.training.train_eval --cats noEFT --masses 500 800          # default: proc:0.5, drop < 100 events
python -m tttg.training.train_eval --scheme baseline                        # old behaviour (all events equal)
python -m tttg.training.train_eval --scheme proc:1                          # fully physical composition
python -m tttg.training.train_eval --scheme proc:0.25                       # any exponent
python -m tttg.training.train_eval --keep-low-stat                          # also train on low-statistics processes
python -m tttg.training.train_eval --min-events 300                         # raise/lower the statistics cut
python -m tttg.training.train_eval --seed 3                                 # different random seed
```
Defaults live in `tttg/config.py` (`TRAIN_SCHEME`, `TRAIN_MIN_EVENTS`, `TRAIN_SEED`); the command-line
options override them. `--min-events` and `--keep-low-stat` are mutually exclusive.

Scheme meaning: `baseline` = every kept background event weight 1. `proc:<a>` = a background process with yield Y
(= per-event physical weight x its number of training events) gets total weight Y^a, spread uniformly over its
events. a=1 physical composition, a=0 all processes equal, a=0.5 tempered (default).

Each run now writes `models/<cat>/Signal_<m>.json` next to the `.pth` (scheme, min_events, seed, dropped
processes, events per process, best epoch, best val AUC, test AUC, weighted test AUC).
**Retraining overwrites `models/<cat>/Signal_<m>.pth`** – back up `models/` first if you want to keep the old ones.
Re-run the downstream steps (NN histograms, limits) after retraining.

## 3. Change list (file by file)

### `tttg/config.py`
* Added `TRAIN_SCHEME = "proc:0.5"`, `TRAIN_MIN_EVENTS = 100`, `TRAIN_SEED = 0`.
  *Why:* you asked for proc:0.5 as default with the scheme and the low-statistics cut being options. The cut value
  100 is the one used in the diagnostics. The seed makes runs reproducible.

### `tttg/data/weights.py` (new functions appended at the end; nothing existing changed)
* `parse_scheme(scheme)` – validates `baseline` / `proc:<a>`. *Why:* a typo must fail immediately, not after hours of training.
* `physical_event_weight(metadata, sample)` – `xsec * LUMINOSITY * 2 / nevents`. *Why:* same number the limit
  histograms use (the 2 because only the test half is used there), so training and limit agree.
  The function takes the metadata dict (no coffea access), so it can be tested alone.
* `process_weights(scheme, n_train, phys_w, min_events)` – per-event weight of each background process
  (yield^a / n; 0 for processes below `min_events`). *Why:* this is the `proc:<a>` recipe from `diagnostics/weighted_training.py`.
* `event_weights(sid, y, per_sample, names, ref)` – expands the process weights to events, gives the signal one
  common weight so that signal and background have equal total weight, and normalises to mean 1 over the
  training set (validation reuses the training factors through `ref`). *Why:* replaces `pos_weight` for class balance and keeps
  the loss at its usual scale so learning rate/OneCycle behave as before.

### `tttg/data/features.py`
* New `TrainingData` dataclass. *Why:* one object carries arrays, weights and bookkeeping instead of a growing tuple.
* `_prepare_data` now also returns `sid`, the sample index of every row. *Why:* needed to give every process its own weight.
* `get_signal_background` carries `sid` through the train/val split and stores `sample_names`, `sid_train/val/test`.
  **Return values, the split (`random_state=42`, stratified) and the scaler are identical to before** (checked on test data,
  `np.array_equal` on all six arrays). *Why:* models trained earlier stay compatible with `get_test_with_sample` and the histogram step,
  and the scaler used at inference does not depend on `--scheme` / `--min-events`.
* New `get_training_data(signal, cat, scheme, min_events)` – builds `TrainingData`: removes low-statistics processes from the
  **train and validation** sets only (they stay in the test set and in the limit histograms), computes loss weights, and
  the physical weight of every test row. *Why:* implements the option set you asked for.
* `get_loaders` got optional `w_train`, `w_val`, `seed`. Without weights it behaves exactly as before. With weights batches are
  `(X, y, w)` and the train loader shuffles every epoch with a seeded generator. *Why:* the old loader used `shuffle=False`
  (order only mixed by the one-off split); the diagnostics shuffled and their results are what this is based on.

### `tttg/training/trainer.py`
* Detects weighted batches (`(X, y, w)`). In weighted mode: loss = mean(BCE per event x w), no `pos_weight`, and the train/val
  AUC (also the early-stopping criterion) is the **weighted AUC**. Unweighted batches go through the old code path.
  *Why:* the model selection must prefer the model that is good for the physical composition, not for the raw one.
* `best_epoch` stored on the trainer. *Why:* goes into the `.json` settings file.

### `tttg/training/train_eval.py`
* New CLI options `--scheme`, `--min-events`, `--keep-low-stat`, `--seed`; `main()` uses `get_training_data`.
* `train()` seeds torch/numpy before building the model and returns a small dict (best val AUC, best epoch, epochs run).
* `test()` takes `test_weights` and additionally prints the cross-section weighted AUC; `_test_plotter` adds the weighted
  background shape and weighted ROC curve to `nn_performance_<m>.png`. Existing numbers/plots are kept.
  *Why:* the unweighted test AUC does not reflect what the limit sees; the weighted one does.
* `write_settings()` writes the `.json` next to the model. *Why:* traceability – a `.pth` alone does not say how it was trained.
* `X_batch, _ = next(iter(loader))` changed to `next(iter(loader))[0]` because batches can now have 3 elements.

### `README.md`
* Step-1 row of the pipeline table lists the new options and points here.

## 4. Differences from the diagnostics experiment (to keep in mind when comparing numbers)
* Low-statistics processes are **removed** from train/val (diagnostics kept them with weight 0). The effect is limited to
  batch composition / batch-norm statistics, negligible for a few events.
* Training uses seed 0 by default; the diagnostics used several seeds.

## 5. How it was tested
No `output.coffea` / coffea in the development sandbox, so the tests used a synthetic coffea-like file (5 processes incl. one with ~56
training events). Verified: (a) split and scaler identical to the original code; (b) per-process total weight ∝ yield^0.5
and signal/background total weight equal; (c) the low-stat process is dropped by default and kept with `--keep-low-stat`;
(d) both schemes and a bad scheme (clear error); (e) same seed ⇒ identical results; (f) old unweighted loaders still train;
(g) `.json` and plots are written; (h) `get_test_with_sample` works. **The real training was not run** – please run it on your data.

## 6. Suggestion (not implemented)
Given the large seed-to-seed spread of the limits, train several seeds per mass and keep the one with the best *validation*
weighted AUC (not the test one). Easy to add on request.


## 7. 2-fold cross-training: every selected event enters the limit

**Why:** the limit used only the TEST half (weight `xsec*lumi*2/N_gen`), so every MC event counted twice as much and the
background MC-stat error was sqrt(2) larger than necessary. Now the events are cut in two halves (part 0 = first
`int(n/2)` events, part 1 = the rest) and two sets of networks are trained: **fold f is trained on the half that is NOT f and
applied to half f**. Every event is scored by a network that never saw it, so ALL events fill the histograms with weight
`xsec*lumi/N_gen` (no factor 2). Fold 0 is exactly the old setup (train = second half, test = first half).

* `tttg/config.py`: `N_FOLDS`; `model_path(cat, m, seed, fold)` -> `Signal_<m>_fold<f>_seed<k>.pth`; `model_paths(cat, m, fold)`
  returns only that fold's members (error if none, never another fold's net). `fold=None` keeps the old behaviour.
* `tttg/data/features.py`: `concatenate_features / get_signal_background / get_training_data / get_test_with_sample` got `fold=0`
  (default = old behaviour); scaler is fitted on the training half of that fold; new `get_folds_with_sample()`
  -> `{sample: [X_part0, X_part1]}`, each part scaled with its own fold's scaler.
* `tttg/data/weights.py`: `WeightManager.sum_genweight / variation` got `use="test"` (old: half, factor 2) or `use="all"`
  (all events, factor 1).
* `tttg/histograms/nn_histograms.py`: `get_histograms` takes `get_folds_with_sample()` output; part f is scored by the ensemble of
  fold f, scores are put back in file order, weights with `use="all"`. The old input raises a clear error.
* `tttg/training/trainer.py`, `train_eval.py`: `fold` passed through (model/plot/json names carry `_fold<f>`); new option
  `--folds` (default `0 1`).
* `limits/pyhf_limits.py`, `limits/root_files.py`, `histograms/plotter.py`: call `get_folds_with_sample` instead of `get_test_with_sample`.
* NOT changed: the NN-loss weights (`physical_event_weight` still has the 2; it is a common factor that cancels in the
  normalisation), `diagnostics/*` (they still work on the old fold-0 half; `yield_stat_from_extractor.py` and the scripts that call
  `get_histograms` need the same one-line switch).
* Old ensembles `Signal_<m>_seed<k>.pth` are fold-0 networks: `for f in models/*/Signal_*_seed*.pth; do mv "$f" "${f/_seed/_fold0_seed}"; done`
  lets you reuse them and train only `--folds 1`.
