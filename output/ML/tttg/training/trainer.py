"""Training loop (AdamW + OneCycleLR, early stopping on validation AUC)."""
import numpy as np
from tqdm import tqdm

import torch
from torch import nn

from sklearn.metrics import roc_auc_score

from tttg.config import EPOCHS, model_path

class OptimizedTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Weighted mode: the datasets hold (X, y, w). The weights already balance signal and background
        # (features.event_weights), so no pos_weight is used and the loss is a weighted mean.
        # Unweighted mode (datasets of (X, y), the old behaviour): class balance through pos_weight.
        self.weighted = len(train_loader.dataset.tensors) == 3

        # Calculate class weights for loss
        self.pos_weight = torch.tensor([(len(train_loader.dataset) - train_loader.dataset.tensors[1].sum()) / 
                                        train_loader.dataset.tensors[1].sum()]).to(self.device)
        
        # Different optimizer choices - try AdamW first
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.002)
        
        # Scheduler with warmup
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.002,
            steps_per_epoch=len(self.train_loader),
            epochs=EPOCHS,
            pct_start=0.3  # 30% warmup
        )

        
        if self.weighted:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")     # multiplied by w in the loops
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def _unpack(self, batch):
        """(X, y, w) -> tensors on the device; w is None in unweighted mode."""
        batch = [b.to(self.device) for b in batch]
        return (batch[0], batch[1], batch[2]) if self.weighted else (batch[0], batch[1], None)

    def _loss(self, outputs, y, w):
        if self.weighted:
            return (self.criterion(outputs, y) * w).mean()
        return self.criterion(outputs, y)

    @staticmethod
    def _auc(labels, preds, weights):
        """AUC; with weights it is the weighted AUC (the same weights as in the loss)."""
        return roc_auc_score(labels, preds, sample_weight=weights if weights else None)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_w = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            batch_X, batch_y, batch_w = self._unpack(batch)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self._loss(outputs, batch_y, batch_w)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                if self.weighted:
                    all_w.extend(batch_w.cpu().numpy().ravel())
        
        avg_loss = total_loss / len(self.train_loader)
        auc = self._auc(all_labels, all_preds, all_w)
        
        return avg_loss, auc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_w = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch_X, batch_y, batch_w = self._unpack(batch)
                outputs = self.model(batch_X)
                loss = self._loss(outputs, batch_y, batch_w)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                if self.weighted:
                    all_w.extend(batch_w.cpu().numpy().ravel())
        
        avg_loss = total_loss / len(self.val_loader)
        auc = self._auc(all_labels, all_preds, all_w)
        
        return avg_loss, auc, all_preds, all_labels
    
    def train(self, epochs=EPOCHS, patience=10, mass="Signal_500", cat="noEFT", seed=None, fold=None):
        best_val_auc = 0
        best_epoch = 0
        best_model_state = None
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        train_aucs = []
        val_aucs = []
        
        for epoch in range(epochs):
            # Training
            train_loss, train_auc = self.train_epoch()
            
            # Validation
            val_loss, val_auc, val_preds, val_labels = self.validate()
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_aucs.append(train_auc)
            val_aucs.append(val_auc)
            
            # Update scheduler
            # self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                # clone the tensors: state_dict().copy() is only a shallow copy, so it would keep
                # pointing at the live weights (and BatchNorm buffers) and end up as the LAST epoch
                best_model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  --> New best model! AUC={best_val_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and save it
        self.model.load_state_dict(best_model_state)
        # `mass` is the signal name, e.g. "Signal_500"
        path = model_path(cat, mass.removeprefix("Signal_"), seed, fold)     # seed/fold given: ensemble member file
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
        self.best_epoch = best_epoch       # read by train_eval for the settings file next to the model
        return best_val_auc, train_losses, val_losses, train_aucs, val_aucs