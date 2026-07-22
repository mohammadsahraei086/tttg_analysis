import numpy as np
from coffea.util import load
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PrepareFeaturesForTraning:
    def __init__(self, output_path: "str" = "../output.coffea"):
        self.output = load(output_path)
        self.training_features = [
            'lepton_pt', 'lepton_eta', 'jet_pt', 'jet_eta', 'bjet_pt', 'met_pt',
            'met_eta', 'ht_goodJets', 'm_wt', 'W_pt', 'top_mass', 'top_pt', 'top_eta', 't_mass', 't_pt',
            't_eta', 'delta_r_ljet', 'delta_r_wl', 'delta_r_wjet', 'delta_r_topl', 'delta_r_topjet', 'delta_r_tl',
            'delta_r_tjet', 'delta_phi_wjet', 'delta_phi_wbjet', 'delta_phi_topjet', 'delta_phi_tjet', 'njets', 'nbjets'
        ]

        self.scaler = StandardScaler()
        self.train, self.test = self.concatenate_features()

    def concatenate_features(self):
        train = {}
        test = {}
        for smpl in self.output["features"]:
            np_features =  np.concatenate(
                [
                    self.output["features"][smpl][var].value.astype(np.float32) 
                    if isinstance(self.output["features"][smpl][var].value[0], np.ndarray)
                    else self.output["features"][smpl][var].value[:, None].astype(np.float32) 
                    for var in self.training_features
                ], 
                axis=1
            )
            length = len(np_features)
            train[smpl] = np_features[int(length/2):]
            test[smpl] = np_features[:int(length/2)]
            
        return train, test

    def get_signal_background(self, signal: str = "Signal_500"):
        
        X_train_raw, y_train_raw = self._prepare_data(self.train, signal)
        
        # Process test data  
        X_test, y_test = self._prepare_data(self.test, signal)
        
        # Split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_raw, y_train_raw, test_size=0.2, 
            random_state=42, stratify=y_train_raw
        )
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.fitted_signal = signal
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _prepare_data(self, np_features, signal):
        """Helper to extract signal and background"""
        signal_data = np_features[signal]
        signal_labels = np.ones(len(signal_data), dtype=np.float32)
        
        background_data_list = [v for k, v in np_features.items() 
                               if not k.startswith('Signal_')]
        background_data = np.vstack(background_data_list)
        background_labels = np.zeros(len(background_data), dtype=np.float32)
        
        X = np.vstack([signal_data, background_data])
        y = np.concatenate([signal_labels, background_labels])
        
        return X, y


    def get_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1))

        # Calculate class weights for sampling
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False) #, sampler=sampler)  # Balanced batches
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_test_with_sample(self, signal="Signal_500"):
        test = {}
        for smpl in self.test:
            try:
                if not self.fitted_signal == signal:
                    raise Exception(f"Scaler Is Not Fitted on {signal}")
                test[smpl] = self.scaler.transform(self.test[smpl])
            except:
                print(f"Scaler Is not Fitted on {signal}, Or Is not Fitted At All")
                print("Trying To Fit Scaler again")
                self.get_signal_background(signal)
                test[smpl] = self.scaler.transform(self.test[smpl])
        
        return test

    def get_total_samples(self, signal="Signal_500"):
        total = {}
        for smpl in self.test:
            try:
                if not self.fitted_signal == signal:
                    raise Exception(f"Scaler Is Not Fitted on {signal}")
                total[smpl] = self.scaler.transform(np.vstack([self.test[smpl], self.train[smpl]]))
            except:
                print(f"Scaler Is not Fitted on {signal}, Or Is not Fitted At All")
                print("Trying To Fit Scaler again")
                self.get_signal_background(signal)
                total[smpl] = self.scaler.transform(np.vstack([self.test[smpl], self.train[smpl]]))
        
        return total