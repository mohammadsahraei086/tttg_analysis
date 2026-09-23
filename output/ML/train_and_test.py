import numpy as np
from typing import Optional
from torch import Tensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch

from preparing_features import PrepareFeaturesForTraning
from simple_classifier import OptimizedHEPClassifier, OptimizedTrainer

def _train_plotter(
    m: int,
    train_losses:  list[float], val_losses:  list[float],
    train_aucs:  list[float], val_aucs:  list[float],
    cat: str="noEFT"
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
    axes[1].axhline(y=0.8359, color='r', linestyle='--', label='RF Benchmark')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title(f'Training and Validation AUC For Siganl_{m}')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{cat}/training_curves_{m}.png', dpi=150)

def train(  
    m: int,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    cat: str = "noEFT"
) -> Optional[tuple[float, ...]]:
    
    X_batch, _ = next(iter(train_loader))
    input_dim = X_batch.shape[1]

    model = OptimizedHEPClassifier(input_dim=input_dim)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = OptimizedTrainer(
        model,
        train_loader,
        val_loader,
    )

    best_auc, train_losses, val_losses, train_aucs, val_aucs = trainer.train(
        epochs=50,
        patience=15,
        mass=f"Signal_{m}",
        cat=cat,
    )

    _train_plotter(
        m,
        train_losses,
        val_losses,
        train_aucs,
        val_aucs,
        cat,
    )

def _test_plotter(
    m: int,
    test_preds:  list[float],
    test_labels:  list[float],
    ks_stat: float,
    test_auc: float,
    cat: str="noEFT"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(test_preds[test_labels==0], bins=50, alpha=0.5, label='Background', density=True)
    axes[0].hist(test_preds[test_labels==1], bins=50, alpha=0.5, label='Signal', density=True)
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
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve Signal_{m}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{cat}/nn_performance_{m}.png', dpi=150)

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
) -> Optional[tuple[float, ...]]:
    print("\n" + "="*50)
    print(f"FINAL TEST EVALUATION for Mass {m}")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_batch, _ = next(iter(test_loader))
    input_dim = X_batch.shape[1]
    
    model = OptimizedHEPClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(f"saved_models/{cat}/Signal_{m}.pth", weights_only=True))

    model.to(device)
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            # batch_y = batch_y.to(device)
            outputs = model(batch_X).to(device)
            probs = torch.sigmoid(outputs)
            test_preds.extend(probs.cpu().numpy())
            test_labels.extend(batch_y.numpy())
    
    test_preds = np.array(test_preds).flatten()
    test_labels = np.array(test_labels).flatten()
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, test_preds_binary)
    test_prec = precision_score(test_labels, test_preds_binary)
    test_rec = recall_score(test_labels, test_preds_binary)
    test_f1 = f1_score(test_labels, test_preds_binary)

    ks_stat, ks_pvalue, sig_scores, bkg_scores = compute_ks_separation(test_preds, test_labels)

    _test_plotter(
        m,
        test_preds,
        test_labels,
        ks_stat,
        test_auc,
        cat
    )
    
    print(f"AUC:       {test_auc:.4f}")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"KS (sig vs bkg): stat={ks_stat:.4f}, p-value={ks_pvalue:.4e}")


if __name__ == "__main__":
    dataset = PrepareFeaturesForTraning()

    masses = [500, 800, 1000, 1200, 1500, 2000, 2500, 3000]
    cats = ["noEFT", "sEFT", "s_hatEFT"]

    for cat in cats:
        for m in masses:
            data = dataset.get_signal_background(signal=f"Signal_{m}", cat=cat)
            train_loader, val_loader, test_loader = dataset.get_loaders(*data)
            train(m, train_loader, val_loader, cat)
            test(m, test_loader, cat)


    
    
    