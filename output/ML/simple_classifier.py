import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
# import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# class FeatureTokenizer(nn.Module):
#     """Turns a (batch, num_features) tensor into (batch, num_features, d_model) tokens."""
#     def __init__(self, num_features, d_model):
#         super().__init__()
#         self.num_features = num_features
#         self.d_model = d_model
#         # one weight vector + bias per feature
#         self.weight = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
#         self.bias = nn.Parameter(torch.zeros(num_features, d_model))

#     def forward(self, x):
#         # x: (batch, num_features)
#         # -> (batch, num_features, 1) * (num_features, d_model) broadcast -> (batch, num_features, d_model)
#         tokens = x.unsqueeze(-1) * self.weight + self.bias
#         return tokens

# class CLSToken(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

#     def forward(self, x):
#         # x: (batch, seq_len, d_model)
#         batch_size = x.size(0)
#         cls_tokens = self.cls.expand(batch_size, -1, -1)  # (batch, 1, d_model)
#         return torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)
        

# class OptimizedHEPClassifier(nn.Module):
#     def __init__(self, num_features=39, d_model=64, nhead=8, num_layers=3, dim_feedforward=128, dropout=0.3):
#         super().__init__()

#         self.tokenizer = FeatureTokenizer(num_features, d_model)
#         self.cls_token = CLSToken(d_model)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             activation='gelu',
#             batch_first=True   # critical, as discussed
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 64),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 1)
#         )

#     def forward(self, x):
#         # x: (batch, num_features)
#         tokens = self.tokenizer(x)              # (batch, num_features, d_model)
#         tokens = self.cls_token(tokens)          # (batch, num_features+1, d_model)
#         encoded = self.transformer(tokens)       # (batch, num_features+1, d_model)
#         cls_output = encoded[:, 0, :]            # take the CLS token's output
#         return self.classifier(cls_output)


class OptimizedHEPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

    # def __init__(self, input_dim):
    #     super().__init__()
    #     self.network = nn.Sequential(
    #         nn.Linear(input_dim, 64),
    #         nn.ReLU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(64, 32),
    #         nn.ReLU(),
    #         nn.Dropout(0.5),
    #         nn.Linear(32, 1),
    #         # nn.Sigmoid()  # For binary classification
    #     )
    
    # def forward(self, x):
    #     return self.network(x)



class OptimizedTrainer:
    def __init__(self, model, device, train_loader, val_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Calculate class weights for loss
        self.pos_weight = torch.tensor([(len(train_loader.dataset) - train_loader.dataset.tensors[1].sum()) / 
                                        train_loader.dataset.tensors[1].sum()]).to(device)
        
        # Different optimizer choices - try AdamW first
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.002)
        
        # Scheduler with warmup
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.002,
            steps_per_epoch=len(self.train_loader),
            epochs=50,
            pct_start=0.3  # 30% warmup
        )

        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_X, batch_y in tqdm(self.train_loader, desc="Training"):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(self.val_loader, desc="Validation"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds)
        
        return avg_loss, auc, all_preds, all_labels
    
    def train(self, epochs=50, patience=10, mass="Signal_500"):
        best_val_auc = 0
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
            self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"Train Loss={train_loss:.4f}, Train AUC={train_auc:.4f} | "
                  f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"  --> New best model! AUC={best_val_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and save it
        self.model.load_state_dict(best_model_state)
        torch.save(self.model.state_dict(), f'saved_models/{mass}.pth')
        
        return best_val_auc, train_losses, val_losses, train_aucs, val_aucs