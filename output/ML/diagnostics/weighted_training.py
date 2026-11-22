"""Does training with physical (cross-section) weights give a better classifier than training on raw events?

    python -m diagnostics.weighted_training --cat noEFT --masses 500 1500
    python -m diagnostics.weighted_training --cat noEFT --masses 500 1500 --schemes baseline proc:0.5 proc:1 --seeds 3

Run from the project root. Nothing in tttg/ is changed and models/ is NOT touched: the experimental
models go to models_experiments/<scheme>/<cat>/Signal_<m>_seed<k>.pth and a summary to
results/diagnostics/weighted_training_<cat>.json.

For every mass and scheme it trains the same network as tttg/ (OptimizedHEPClassifier, AdamW, OneCycleLR
stepped per batch, grad clip 1, batch 256, same train/val split, same early-stopping patience) and only
changes WHAT THE LOSS (and the early-stopping AUC) SEES. Schemes:

  baseline     what the analysis does now: every event counts the same, classes balanced
  proc:<a>     every background PROCESS contributes to the loss in proportion to (its yield)^a,
               spread uniformly over its events. a=1 is the physical composition (per-event weight =
               xsec*lumi/N_gen), a=0.5 is tempered, a=0 gives every process the same total weight.
               Processes with fewer than --min-events training events get weight 0 in the loss (they
               are still evaluated); the early-stopping AUC uses the same weights.
  (the signal always has uniform weight; signal and background are balanced in every scheme)

Every model is evaluated on the TEST half with the physical weights, exactly as in weighted_performance:

  auc_w   class-normalised weighted AUC, signal vs all background (physical composition)
  auc_tt  AUC signal vs tt only (tt is ~90% of the yield)
  z_shape shape significance sqrt(sum_bins S^2/B) over score bins merged from the top until the background
          has N_eff >= 11 (the same rule as the limit rebinning). The closest cheap proxy to the limit
  gain    best cut-scan S/sqrt(B) divided by the no-cut value (noisy: it is a maximum over cuts)
  auc_u   plain unweighted AUC (only to show how the old metric behaves)

Several seeds are trained per scheme and mean +- std is printed: a difference smaller than the seed-to-seed
spread is NOT a real difference. The model currently in models/ is evaluated as a reference.
This is a sensitivity proxy, NOT a limit: if one scheme wins clearly, confirm it with the pyhf limit.
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tttg.config import CATS, EPOCHS, MASSES, MODEL_DIR, PATIENCE, RESULTS_DIR, model_path

EXP_DIR = MODEL_DIR.parent / "models_experiments"
OUT_DIR = RESULTS_DIR / "diagnostics"
SILENT = lambda *a, **k: None          # noqa: E731


# ----------------------------------------------------------------------------- weights
def process_weights(scheme, n_train, phys_w, min_events=100):
    """Per-event background weight of every process for a scheme.
    n_train: {sample: training events}, phys_w: {sample: physical weight per event}."""
    if scheme == "baseline":
        return {s: 1.0 for s in n_train}
    if not scheme.startswith("proc:"):
        raise ValueError(f"unknown scheme {scheme!r} (use 'baseline' or 'proc:<alpha>')")
    alpha = float(scheme.split(":", 1)[1])
    out = {}
    for s, n in n_train.items():
        if n < min_events:
            out[s] = 0.0
        else:
            out[s] = (phys_w[s] * n) ** alpha / n          # process total ~ yield^alpha, uniform per event
    return out


def event_weights(sid, y, per_sample, names, ref=None):
    """Per-event weights for the rows of one set (train or val).  Signal rows get 1 first; then signal and
    background are balanced and the mean over the TRAINING set is 1 (ref = factors from the training set)."""
    w = np.ones(len(y), dtype=np.float64)
    bkg = y == 0
    lut = np.array([per_sample.get(n, 1.0) for n in names])
    w[bkg] = lut[sid[bkg]]
    if ref is None:
        if w[bkg].sum() <= 0:
            raise ValueError("all background weights are zero (lower --min-events)")
        sig_scale = w[bkg].sum() / (~bkg).sum()
        w[~bkg] = sig_scale
        norm = w.mean()
        ref = (sig_scale, norm)
    else:
        w[~bkg] = ref[0]
    return w / ref[1], ref


# ----------------------------------------------------------------------------- metrics
def shape_significance(scores, weights, signal, min_neff=11, nfine=40):
    """sqrt(sum_bins S^2/B); fine score bins are merged from the top until N_eff(B) >= min_neff."""
    bkgs = [b for b in scores if b != signal]
    edges = np.linspace(0, 1, nfine + 1)
    S = np.histogram(scores[signal], bins=edges, weights=weights[signal])[0]
    B = sum(np.histogram(scores[b], bins=edges, weights=weights[b])[0] for b in bkgs)
    B2 = sum(np.histogram(scores[b], bins=edges, weights=weights[b] ** 2)[0] for b in bkgs)
    closed, s, b, b2 = [], 0.0, 0.0, 0.0
    for i in range(nfine - 1, -1, -1):
        s, b, b2 = s + S[i], b + B[i], b2 + B2[i]
        if b > 0 and b ** 2 / b2 >= min_neff:
            closed.append([s, b])
            s = b = b2 = 0.0
    if not closed:
        return float("nan")
    closed[-1][0] += s                    # leftover low-score tail joins the last bin
    closed[-1][1] += b
    return float(np.sqrt(sum(s ** 2 / b for s, b in closed)))


def metrics(scores, weights, signal):
    from diagnostics.weighted_performance import evaluate
    r = evaluate(scores, weights, signal, out=SILENT)
    scan = r["scan"]
    return dict(auc_u=r["auc"]["unweighted"], auc_w=r["auc"]["weighted"], auc_tt=r["auc"].get("tt", np.nan),
                z_shape=shape_significance(scores, weights, signal),
                gain=(scan["z"] / scan["z_nocut"]) if scan else np.nan)


# ----------------------------------------------------------------------------- data
def build(ds, cat, signal):
    """Same arrays and the same train/val split as PrepareFeaturesForTraning.get_signal_background,
    but keeping track of which sample every row comes from."""
    train, test = ds.concatenate_features(signal, cat)
    names = [signal] + [k for k in train if not k.startswith("Signal_")]
    X = np.vstack([train[n] for n in names])
    y = np.concatenate([np.ones(len(train[signal])), np.zeros(len(X) - len(train[signal]))]).astype(np.float32)
    sid = np.concatenate([np.full(len(train[n]), i) for i, n in enumerate(names)])
    Xtr, Xva, ytr, yva, str_, sva = train_test_split(X, y, sid, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    return dict(names=names, Xtr=scaler.transform(Xtr), ytr=ytr, sid_tr=str_,
                Xva=scaler.transform(Xva), yva=yva, sid_va=sva,
                test={n: scaler.transform(x) for n, x in test.items()},
                n_train={n: int(len(train[n])) for n in names})


# ----------------------------------------------------------------------------- training
def fit(d, wtr, wva, seed, epochs, patience, save_path=None):
    """Train one network; early stopping on the validation AUC computed with the validation weights."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from tttg.training.model import OptimizedHEPClassifier

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = OptimizedHEPClassifier(d["Xtr"].shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.002)
    ds = TensorDataset(torch.FloatTensor(d["Xtr"]), torch.FloatTensor(d["ytr"]).reshape(-1, 1),
                       torch.FloatTensor(wtr).reshape(-1, 1))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=gen)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.002, steps_per_epoch=len(loader),
                                                epochs=epochs, pct_start=0.3)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    Xva = torch.FloatTensor(d["Xva"]).to(dev)

    best, best_state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()                       # per batch, as in the (fixed) trainer
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xva)).cpu().numpy().ravel()
        auc = roc_auc_score(d["yva"], p, sample_weight=wva)
        if auc > best:
            best, bad, best_ep = auc, 0, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model, best, best_ep, dev


def predict(model, dev, test):
    import torch
    out = {}
    with torch.no_grad():
        for s, x in test.items():
            out[s] = torch.sigmoid(model(torch.FloatTensor(x).to(dev))).cpu().numpy().ravel()
    return out


def reference_scores(d, mass, cat):
    """Scores of the model currently in models/ (same scaler as the training split)."""
    import torch
    from tttg.training.model import OptimizedHEPClassifier
    path = model_path(cat, mass)
    if not path.exists():
        return None
    model = OptimizedHEPClassifier(d["Xtr"].shape[1])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return predict(model, torch.device("cpu"), d["test"])


# ----------------------------------------------------------------------------- reporting
KEYS = ["auc_w", "auc_tt", "z_shape", "gain", "auc_u"]


def summarise(rows):
    """rows: list of metric dicts (one per seed) -> {key: (mean, std)}"""
    return {k: (float(np.nanmean([r[k] for r in rows])),
                float(np.nanstd([r[k] for r in rows])) if len(rows) > 1 else 0.0) for k in KEYS}


def print_table(mass, table, out=print):
    out(f"\n  Signal_{mass}   (mean +- std over seeds; a difference below the std is not real)")
    out(f"  {'scheme':<14}{'seeds':>6}" + "".join(f"{k:>16}" for k in KEYS) + f"{'val ep':>8}")
    for name, (summ, nseed, ep) in table.items():
        cells = "".join(f"{summ[k][0]:>10.4f}+-{summ[k][1]:<5.4f}" if summ[k][1] else f"{summ[k][0]:>16.4f}" for k in KEYS)
        out(f"  {name:<14}{nseed:>6}{cells}{ep:>8.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cat", choices=CATS, default="noEFT")
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=[500, 1500])
    p.add_argument("--schemes", nargs="+", default=["baseline", "proc:0.5", "proc:1"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--min-events", type=int, default=100,
                   help="processes with fewer training events get weight 0 in the weighted schemes")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--no-reference", action="store_true", help="skip the model currently in models/")
    a = p.parse_args()

    from diagnostics.common import load_weights
    from tttg.data.features import PrepareFeaturesForTraning

    ds = PrepareFeaturesForTraning()
    print(f"{len(a.masses)} masses x {len(a.schemes)} schemes x {a.seeds} seeds = "
          f"{len(a.masses) * len(a.schemes) * a.seeds} trainings of up to {a.epochs} epochs")
    summary = {}
    for m in a.masses:
        signal = f"Signal_{m}"
        d = build(ds, a.cat, signal)
        names = d["names"]
        weights = load_weights(a.cat, {n: x[:, 0] for n, x in d["test"].items()})   # physical test weights
        phys = {n: float(weights[n][0]) for n in names if n != signal}
        n_bkg = {n: d["n_train"][n] for n in names if n != signal}
        table = {}
        if not a.no_reference:
            sc = reference_scores(d, m, a.cat)
            if sc is not None:
                table["models/ (now)"] = ({k: (v, 0.0) for k, v in metrics(sc, weights, signal).items()}, 1, np.nan)
        for scheme in a.schemes:
            per = process_weights(scheme, n_bkg, phys, a.min_events)
            per_full = {signal: 1.0, **per}
            wtr, ref = event_weights(d["sid_tr"], d["ytr"], per_full, names)
            wva, _ = event_weights(d["sid_va"], d["yva"], per_full, names, ref)
            used = [n for n, v in per.items() if v > 0]
            print(f"\n[{signal}] scheme {scheme}: {len(used)}/{len(per)} background processes in the loss"
                  + ("" if len(used) == len(per) else f" (dropped: {', '.join(n for n in per if per[n] == 0)})"))
            rows, eps = [], []
            for seed in range(a.seeds):
                tag = scheme.replace(":", "_")
                model, best, ep, dev = fit(d, wtr, wva, seed, a.epochs, PATIENCE,
                                           EXP_DIR / tag / a.cat / f"{signal}_seed{seed}.pth")
                rows.append(metrics(predict(model, dev, d["test"]), weights, signal))
                eps.append(ep)
                print(f"   seed {seed}: best val AUC {best:.4f} at epoch {ep};  "
                      + "  ".join(f"{k} {rows[-1][k]:.4f}" for k in KEYS[:3]))
            table[scheme] = (summarise(rows), a.seeds, float(np.mean(eps)))
        print_table(m, table)
        summary[signal] = {k: {kk: list(vv) for kk, vv in v[0].items()} for k, v in table.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    f = OUT_DIR / f"weighted_training_{a.cat}.json"
    f.write_text(json.dumps(summary, indent=1))
    print(f"\nsummary written to {f}")


if __name__ == "__main__":
    main()
