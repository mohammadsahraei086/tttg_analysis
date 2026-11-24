"""Train and evaluate one NN per (category, signal mass); writes models/ and plots/training/."""
import json
import numpy as np
from typing import Optional
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

from tttg.config import (CATS, EPOCHS, MASSES, N_SEEDS, PATIENCE, RF_BENCHMARK_AUC, TRAIN_MIN_EVENTS,
                         TRAIN_PLOT_DIR, TRAIN_SCHEME, TRAIN_SEED, model_path, model_paths)
from tttg.data.features import PrepareFeaturesForTraning
from tttg.training.model import OptimizedHEPClassifier
from tttg.training.trainer import OptimizedTrainer

def _train_plotter(
    m: int,
    train_losses:  list[float], val_losses:  list[float],
    train_aucs:  list[float], val_aucs:  list[float],
    cat: str="noEFT",
    seed: Optional[int] = None,
    fold: Optional[int] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(train_losses, label='Train', linewidth=2)
    axes[0].plot(val_losses, label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training and Validation Loss For Siganl_{m}')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(train_aucs, label='Train', linewidth=2)
    axes[1].plot(val_aucs, label='Validation', linewidth=2)
    axes[1].axhline(y=RF_BENCHMARK_AUC, color='r', linestyle='--', label='RF Benchmark')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title(f'Training and Validation AUC For Siganl_{m}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    tag = ("" if fold is None else f"_fold{fold}") + ("" if seed is None else f"_seed{seed}")
    out = TRAIN_PLOT_DIR / cat / f"training_curves_{m}{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)

def train(  
    m: int,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    cat: str = "noEFT",
    seed: int = TRAIN_SEED,
    fold: Optional[int] = None,
) -> dict:
    """Train ONE ensemble member (saved as Signal_<m>_fold<fold>_seed<seed>.pth); returns a small dict (seed, best
    val AUC, best epoch, epochs run) for the settings file."""
    # seed BEFORE the model is built: initial weights, dropout masks (and, via get_loaders, batch order)
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_batch = next(iter(train_loader))[0]      # batches are (X, y) or (X, y, w)
    input_dim = X_batch.shape[1]

    model = OptimizedHEPClassifier(input_dim=input_dim)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = OptimizedTrainer(
        model,
        train_loader,
        val_loader,
    )

    best_auc, train_losses, val_losses, train_aucs, val_aucs = trainer.train(
        epochs=EPOCHS,
        patience=PATIENCE,
        mass=f"Signal_{m}",
        cat=cat,
        seed=seed,
        fold=fold,
    )

    _train_plotter(
        m,
        train_losses,
        val_losses,
        train_aucs,
        val_aucs,
        cat,
        seed,
        fold,
    )
    return {"seed": int(seed), "best_val_auc": float(best_auc), "best_epoch": int(trainer.best_epoch),
            "epochs_run": len(val_aucs)}

def _test_plotter(
    m: int,
    test_preds:  list[float],
    test_labels:  list[float],
    ks_stat: float,
    test_auc: float,
    cat: str="noEFT",
    test_weights=None,
    test_auc_w: Optional[float] = None,
    fold: Optional[int] = None,
) -> None:
    """test_weights = physical weight of every test row; if given, the physically weighted background
    shape and ROC curve are added (this is the background composition the limit actually sees)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(test_preds[test_labels==0], bins=50, alpha=0.5, label='Background', density=True)
    axes[0].hist(test_preds[test_labels==1], bins=50, alpha=0.5, label='Signal', density=True)
    if test_weights is not None:
        axes[0].hist(test_preds[test_labels==0], bins=50, weights=test_weights[test_labels==0], density=True,
                     histtype='step', color='k', linewidth=1.5, label='Background (xsec-weighted)')
    axes[0].set_xlabel('NN Output Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'Prediction Distributions; Signal_{m}')
    axes[0].text(0.5, 0.95, f'KS = {ks_stat:.4f}',
                 transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_labels, test_preds)
    axes[1].plot(fpr, tpr, linewidth=2, label=f'NN (AUC = {test_auc:.4f})')
    if test_weights is not None:
        fpr_w, tpr_w, _ = roc_curve(test_labels, test_preds, sample_weight=test_weights)
        axes[1].plot(fpr_w, tpr_w, linewidth=2, color='k', label=f'NN xsec-weighted (AUC = {test_auc_w:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve Signal_{m}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = TRAIN_PLOT_DIR / cat / f"nn_performance_{m}{'' if fold is None else f'_fold{fold}'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)

def compute_ks_separation(test_preds, test_labels):
    """KS test between signal and background NN output distributions (separation power)."""
    sig_scores = test_preds[test_labels == 1]
    bkg_scores = test_preds[test_labels == 0]
    
    ks_stat, p_value = ks_2samp(sig_scores, bkg_scores)
    
    return ks_stat, p_value, sig_scores, bkg_scores

def test(
    m: int,
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    cat: str="noEFT",
    test_weights=None,
    paths=None,
    fold: Optional[int] = None,
) -> dict:
    """Evaluate the NN ENSEMBLE (mean of the members' sigmoid outputs) on the test half.

    test_weights: physical weight per test row (TrainingData.w_test, same order as the loader).
    Adds the cross-section weighted AUC (what the limit cares about) next to the usual unweighted numbers.
    paths: the member files (default: every member found by config.model_paths(cat, m, fold)).
    fold: the test set is the half that fold's networks did not see in training."""
    print("\n" + "="*50)
    print(f"FINAL TEST EVALUATION for Mass {m}" + ("" if fold is None else f" (fold {fold})"))
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_batch = next(iter(test_loader))[0]
    input_dim = X_batch.shape[1]

    paths = list(paths) if paths is not None else model_paths(cat, m, fold)
    if not paths:
        raise FileNotFoundError(f"no trained model for {cat} Signal_{m}")
    test_labels = test_loader.dataset.tensors[1].numpy().ravel()       # loader order = dataset order (no shuffle)

    member_preds = []
    for path in paths:
        model = OptimizedHEPClassifier(input_dim=input_dim)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.to(device)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                preds.append(torch.sigmoid(model(batch_X.to(device))).cpu().numpy().ravel())
        member_preds.append(np.concatenate(preds))

    test_preds = np.mean(member_preds, axis=0)                          # the ensemble score
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds_binary)
    test_prec = precision_score(test_labels, test_preds_binary)
    test_rec = recall_score(test_labels, test_preds_binary)
    test_f1 = f1_score(test_labels, test_preds_binary)

    ks_stat, ks_pvalue, sig_scores, bkg_scores = compute_ks_separation(test_preds, test_labels)

    test_auc_w = None
    if test_weights is not None:
        test_weights = np.asarray(test_weights)
        # signal rows have weight 1 (uniform), so only the background composition is weighted
        test_auc_w = roc_auc_score(test_labels, test_preds, sample_weight=test_weights)

    _test_plotter(
        m,
        test_preds,
        test_labels,
        ks_stat,
        test_auc,
        cat,
        test_weights,
        test_auc_w,
        fold,
    )
    
    member_auc_w = None
    if test_weights is not None:
        member_auc_w = [float(roc_auc_score(test_labels, p, sample_weight=test_weights)) for p in member_preds]
        print(f"Ensemble of {len(paths)} model(s); xsec-weighted AUC of each member: "
              + ", ".join(f"{a:.4f}" for a in member_auc_w))
    print(f"AUC:       {test_auc:.4f}")
    if test_auc_w is not None:
        print(f"AUC (xsec-weighted bkg): {test_auc_w:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"KS (sig vs bkg): stat={ks_stat:.4f}, p-value={ks_pvalue:.4e}")
    return {"test_auc": float(test_auc),
            "test_auc_weighted": None if test_auc_w is None else float(test_auc_w),
            "member_test_auc_weighted": member_auc_w,
            "ks": float(ks_stat)}


def write_settings(m, cat, data, seeds, train_infos, test_info, fold):
    """models/<cat>/Signal_<m>_fold<fold>.json: how this ensemble was trained, so a .pth is never a mystery later."""
    out = model_path(cat, m, fold=fold).with_suffix(".json")
    info = {
        "fold": fold, "trained_on_part": 1 - fold, "tested_on_part": fold,
        "scheme": data.scheme, "min_events": data.min_events,
        "seeds": list(seeds), "n_members": len(seeds),
        "members": [model_path(cat, m, s, fold).name for s in seeds],
        "dropped_processes": data.dropped,
        "train_events_per_process": data.n_train,
        "per_event_weight": data.process_weight,
        "n_train_rows_used": int(len(data.y_train)),
        "per_member": train_infos,                      # seed, best val AUC, best epoch, epochs run of each member
        **test_info,                                    # metrics of the ENSEMBLE on the test half
    }
    out.write_text(json.dumps(info, indent=2))
    return out


def main():
    import argparse

    p = argparse.ArgumentParser(description="Train + test the NN for every category and mass.")
    p.add_argument("--cats", nargs="+", choices=CATS, default=CATS)
    p.add_argument("--masses", nargs="+", type=int, choices=MASSES, default=MASSES)
    p.add_argument("--scheme", default=TRAIN_SCHEME,
                   help="background weighting: 'baseline' (all events equal) or 'proc:<a>' (process total "
                        f"~ yield^a; a=1 physical, a=0 all processes equal). default: {TRAIN_SCHEME}")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--min-events", type=int, default=TRAIN_MIN_EVENTS,
                   help="background processes with fewer training events are left out of the training "
                        f"(default {TRAIN_MIN_EVENTS}; 0 keeps all)")
    g.add_argument("--keep-low-stat", action="store_true",
                   help="shortcut for --min-events 0: train on every background process")
    p.add_argument("--seed", type=int, default=TRAIN_SEED,
                   help=f"first seed of the ensemble (default {TRAIN_SEED}); the members use seed, seed+1, ...")
    p.add_argument("--seeds", type=int, default=N_SEEDS,
                   help=f"number of ensemble members = networks trained per (cat, mass) (default {N_SEEDS})")
    p.add_argument("--folds", nargs="+", type=int, choices=[0, 1], default=[0, 1],
                   help="2-fold cross-training: fold f is trained on the half that is NOT f and tested on half f. "
                        "Train both (default) so that every event can enter the limit.")
    args = p.parse_args()
    min_events = 0 if args.keep_low_stat else args.min_events
    seeds = [args.seed + k for k in range(args.seeds)]
    print(f"Training setup: scheme={args.scheme}, min_events={min_events}, ensemble seeds={seeds}")

    dataset = PrepareFeaturesForTraning()
    for cat in args.cats:
        for m in args.masses:
            for fold in args.folds:
                data = dataset.get_training_data(signal=f"Signal_{m}", cat=cat,
                                                 scheme=args.scheme, min_events=min_events, fold=fold)
                if data.dropped:
                    print(f"Left out of the training (< {min_events} training events): {data.dropped}")
                train_infos = []
                for s in seeds:
                    print(f"\n##### {cat} Signal_{m} fold {fold}: ensemble member seed {s} "
                          f"({len(train_infos) + 1}/{len(seeds)})")
                    # the training loader shuffles with a generator seeded by s -> rebuild the loaders per member
                    train_loader, val_loader, test_loader = dataset.get_loaders(
                        data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test,
                        w_train=data.w_train, w_val=data.w_val, seed=s)
                    train_infos.append(train(m, train_loader, val_loader, cat, seed=s, fold=fold))

                members = [model_path(cat, m, s, fold) for s in seeds]
                # members of an earlier, different ensemble (other seeds, SAME fold) would be picked up by the histogram step
                for old in model_paths(cat, m, fold):
                    if old not in members:
                        print(f"removing stale ensemble member {old.name}")
                        old.unlink()
                test_info = test(m, test_loader, cat, test_weights=data.w_test, paths=members, fold=fold)
                write_settings(m, cat, data, seeds, train_infos, test_info, fold)


if __name__ == "__main__":
    main()