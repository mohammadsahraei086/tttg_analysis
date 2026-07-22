"""
Boosted Decision Tree module for High Energy Physics classification
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class BDTTrainer:
    """Train and evaluate Boosted Decision Trees for comparison with neural networks"""
    
    def __init__(self, use_xgboost=True):
        """
        Initialize BDT trainer
        
        Parameters:
        -----------
        use_xgboost : bool
            If True, use XGBoost. If False, use sklearn's GradientBoostingClassifier
        """
        self.use_xgboost = use_xgboost
        self.model = None
        self.feature_importances = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, optimize=True):
        """
        Train BDT model with optional hyperparameter optimization
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data (optional, used for early stopping)
        optimize : bool
            Whether to perform hyperparameter optimization
        """
        
        if self.use_xgboost:
            if optimize:
                # Grid search for XGBoost hyperparameters
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'min_child_weight': [1, 3, 5]
                }
                
                print("Performing hyperparameter optimization for XGBoost...")
                xgb_model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42
                )
                
                grid_search = GridSearchCV(
                    xgb_model, 
                    param_grid, 
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                if X_val is not None and y_val is not None:
                    grid_search.fit(X_train, y_train, 
                                   eval_set=[(X_val, y_val)],
                                   verbose=False)
                else:
                    grid_search.fit(X_train, y_train)
                
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV AUC: {grid_search.best_score_:.4f}")
                
            else:
                # Default XGBoost with good hyperparameters for HEP
                # Note: early_stopping_rounds is now passed in the constructor
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42,
                    early_stopping_rounds=20  # Moved here from fit()
                )
                
                if X_val is not None and y_val is not None:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    self.model.fit(X_train, y_train)
        
        else:
            # Use sklearn's GradientBoostingClassifier
            if optimize:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 5, 6],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.8, 0.9, 1.0],
                    'min_samples_split': [2, 5, 10]
                }
                
                print("Performing hyperparameter optimization for GBDT...")
                gbdt_model = GradientBoostingClassifier(random_state=42)
                
                grid_search = GridSearchCV(
                    gbdt_model,
                    param_grid,
                    cv=3,
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV AUC: {grid_search.best_score_:.4f}")
                
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                )
                self.model.fit(X_train, y_train)
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_preds = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"Validation AUC: {val_auc:.4f}")
            return val_auc
        
        return None
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.use_xgboost:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold=0.5):
        """Get binary predictions"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_metrics(self, X, y_true, threshold=0.5):
        """
        Compute classification metrics
        
        Returns:
        --------
        dict: Dictionary containing AUC, accuracy, precision, recall, F1
        """
        y_pred_proba = self.predict_proba(X)
        y_pred = self.predict(X, threshold)
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model"""
        import joblib
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved model"""
        import joblib
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


class ModelComparator:
    """
    Compare neural network and BDT performance
    """
    
    def __init__(self, feature_names=None, nn_module=None):
        """
        Parameters:
        -----------
        feature_names : list
            List of feature names for plotting
        nn_module : module
            The module containing your neural network classes (OptimizedHEPClassifier, OptimizedTrainer)
        """
        self.feature_names = feature_names
        self.nn_module = nn_module
        self.results = {}
        
    def compare_models(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                       mass="Signal_500", train_nn=True, device=None,
                       bdt_optimize=False):
        """
        Train and compare both models
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data
        X_test, y_test : Test data
        mass : str
            Signal mass label
        train_nn : bool
            Whether to train neural network
        device : torch.device
            Device for neural network training
        bdt_optimize : bool
            Whether to optimize BDT hyperparameters
        
        Returns:
        --------
        dict: Results for both models
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        results = {}
        
        # 1. Train BDT
        print("\n" + "="*50)
        print("TRAINING BDT MODEL")
        print("="*50)
        
        bdt_trainer = BDTTrainer(use_xgboost=True)
        bdt_val_auc = bdt_trainer.train(X_train, y_train, X_val, y_val, optimize=bdt_optimize)
        
        # BDT metrics on test set
        bdt_metrics = bdt_trainer.get_metrics(X_test, y_test)
        results['bdt'] = {
            'val_auc': bdt_val_auc,
            'test_metrics': bdt_metrics,
            'feature_importances': bdt_trainer.feature_importances,
            'trainer': bdt_trainer
        }
        
        print(f"\nBDT Test AUC: {bdt_metrics['auc']:.4f}")
        print(f"BDT Test F1: {bdt_metrics['f1']:.4f}")
        
        # 2. Train Neural Network (if requested)
        if train_nn and self.nn_module is not None:
            print("\n" + "="*50)
            print("TRAINING NEURAL NETWORK MODEL")
            print("="*50)
            
            # Prepare data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train), 
                torch.FloatTensor(y_train).reshape(-1, 1)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val), 
                torch.FloatTensor(y_val).reshape(-1, 1)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
            
            # Train NN
            model = self.nn_module.OptimizedHEPClassifier(input_dim=X_train.shape[1])
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            trainer = self.nn_module.OptimizedTrainer(model, device, train_loader, val_loader)
            best_nn_auc, _, _, _, _ = trainer.train(epochs=50, patience=15, mass=mass)
            
            # Get test predictions
            model.eval()
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test), 
                torch.FloatTensor(y_test).reshape(-1, 1)
            )
            test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
            
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    probs = torch.sigmoid(outputs)
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(batch_y.numpy())
            
            nn_metrics = {
                'auc': roc_auc_score(all_labels, all_preds),
                'accuracy': accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)),
                'precision': precision_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)),
                'recall': recall_score(all_labels, (np.array(all_preds) >= 0.5).astype(int)),
                'f1': f1_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
            }
            
            results['nn'] = {
                'val_auc': best_nn_auc,
                'test_metrics': nn_metrics,
                'model': model
            }
            
            print(f"\nNN Test AUC: {nn_metrics['auc']:.4f}")
            print(f"NN Test F1: {nn_metrics['f1']:.4f}")
        
        elif train_nn and self.nn_module is None:
            print("Warning: nn_module not provided. Skipping neural network training.")
        
        self.results = results
        return results
    
    def print_comparison(self, mass="Signal_500"):
        """Print comparison results"""
        if not self.results:
            print("No results to compare. Run compare_models first.")
            return
        
        print("\n" + "="*60)
        print(f"MODEL COMPARISON FOR {mass}")
        print("="*60)
        
        print("\nPerformance on Test Set:")
        print("-"*40)
        
        for model_name, result in self.results.items():
            metrics = result['test_metrics']
            print(f"\n{model_name.upper()}:")
            print(f"  AUC:        {metrics['auc']:.4f}")
            print(f"  Accuracy:   {metrics['accuracy']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  F1 Score:   {metrics['f1']:.4f}")
            
            if 'val_auc' in result and result['val_auc'] is not None:
                print(f"  Val AUC:    {result['val_auc']:.4f}")
        
        # Determine best model
        if 'nn' in self.results and 'bdt' in self.results:
            nn_auc = self.results['nn']['test_metrics']['auc']
            bdt_auc = self.results['bdt']['test_metrics']['auc']
            
            if nn_auc > bdt_auc:
                print(f"\n✅ Neural Network outperforms BDT by {abs(nn_auc - bdt_auc):.4f} AUC")
            elif bdt_auc > nn_auc:
                print(f"\n✅ BDT outperforms Neural Network by {abs(nn_auc - bdt_auc):.4f} AUC")
            else:
                print("\n⚠️ Both models perform equally")
    
    def plot_feature_importances(self, top_n=20, save_path=None):
        """Plot feature importances from BDT"""
        if 'bdt' not in self.results:
            print("BDT model not found in results")
            return
        
        import matplotlib.pyplot as plt
        
        importances = self.results['bdt']['feature_importances']
        
        if importances is None or self.feature_names is None:
            print("Feature importances or feature names not available")
            return
        
        # FIX: Ensure top_n doesn't exceed number of features
        n_features = len(importances)-11
        if top_n > n_features:
            print(f"Warning: top_n ({top_n}) exceeds number of features ({n_features}). Setting to {n_features}.")
            top_n = n_features
        
        # Sort features by importance (descending order)
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Get the feature names for these indices
        top_feature_names = [self.feature_names[i] for i in indices]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), top_feature_names, rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances (BDT)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    def get_results_summary(self):
        """Get a summary of results as a dictionary"""
        summary = {}
        for model_name, result in self.results.items():
            summary[model_name] = {
                'auc': result['test_metrics']['auc'],
                'f1': result['test_metrics']['f1'],
                'val_auc': result.get('val_auc', None)
            }
        return summary