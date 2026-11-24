"""Feature preparation for the NN training: reads the coffea output, builds train/val/test sets."""
import numpy as np
from coffea.util import load

from dataclasses import dataclass, field

from tttg.config import COFFEA_OUTPUT, TRAIN_MIN_EVENTS, TRAIN_SCHEME, TRAIN_SEED
from tttg.data.weights import event_weights, parse_scheme, physical_event_weight, process_weights
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@dataclass
class TrainingData:
    """Everything train_eval needs for one (signal, cat): arrays, training weights and bookkeeping.

    X_train / X_val contain only the rows that are actually used in the training (low-statistics
    background processes are already removed, see `dropped`). X_test is NEVER filtered: it holds every sample.
    w_train / w_val : loss weights (class-balanced, mean 1 over the training set)
    w_test          : PHYSICAL weight of every test row (background: xsec*lumi*2/N_gen, signal: 1); only used
                      for the weighted performance numbers/plots, never for the training
    """
    X_train: np.ndarray
    y_train: np.ndarray
    w_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    w_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    w_test: np.ndarray
    scheme: str
    min_events: int
    dropped: list = field(default_factory=list)      # background processes left out of the training
    n_train: dict = field(default_factory=dict)      # training events per process (before dropping)
    process_weight: dict = field(default_factory=dict)   # per-event weight of every background process


class PrepareFeaturesForTraning:
    def __init__(self, output_path=COFFEA_OUTPUT):
        self.output = load(str(output_path))
        self.training_features = [
            'lepton_pt', 'lepton_eta', 'jet_pt', 'jet_eta', 'bjet_pt', 'met_pt', "s_t", "s_hat",
            'met_eta', 'ht_goodJets', 'm_wt', 'W_pt', 'top_mass', 'top_pt', 'top_eta', 't_mass', 't_pt',
            't_eta', 'delta_r_ljet', 'delta_r_wl', 'delta_r_wjet', 'delta_r_topl', 'delta_r_topjet', 'delta_r_tl',
            'delta_r_tjet', 'delta_phi_wjet', 'delta_phi_wbjet', 'delta_phi_topjet', 'delta_phi_tjet', 'njets', 'nbjets'
        ]

        self.scaler = StandardScaler()
        self.fitted = None 
        

    @staticmethod
    def _split_halves(arr, fold):
        """(train, test) arrays of one sample for `fold`. The events are cut in two halves in file order:
        part 0 = first int(n/2) events, part 1 = the rest. Fold f is TESTED on part f and TRAINED on the other part.
        fold=0 is exactly the old split (train = second half, test = first half)."""
        h = int(len(arr) / 2)
        parts = (arr[:h], arr[h:])
        return parts[1 - fold].copy(), parts[fold].copy()

    def concatenate_features(self, signal, cat="noEFT", fold=0):
        if fold not in (0, 1):
            raise ValueError(f"fold must be 0 or 1, got {fold!r}")
        train = {}
        test = {}
        np_features =  np.concatenate(
            [
                self.output["features"][cat][signal][var].value.astype(np.float32) 
                if isinstance(self.output["features"][cat][signal][var].value[0], np.ndarray)
                else self.output["features"][cat][signal][var].value[:, None].astype(np.float32) 
                for var in self.training_features
            ], 
            axis=1
        )
        train[signal], test[signal] = self._split_halves(np_features, fold)
        
        for smpl in self.output["features"]["noEFT"]:
            if smpl.startswith("Signal_"):      # NEW: don't overwrite / pull in other signals
                continue
            np_features =  np.concatenate(
                [
                    self.output["features"]["noEFT"][smpl][var].value.astype(np.float32) 
                    if isinstance(self.output["features"]["noEFT"][smpl][var].value[0], np.ndarray)
                    else self.output["features"]["noEFT"][smpl][var].value[:, None].astype(np.float32) 
                    for var in self.training_features
                ], 
                axis=1
            )
            train[smpl], test[smpl] = self._split_halves(np_features, fold)
            
        return train, test

    def get_signal_background(self, signal: str = "Signal_500", cat="noEFT", fold=0):
        self.train, self.test = self.concatenate_features(signal, cat, fold)
        X_train_raw, y_train_raw, sid_train_raw = self._prepare_data(self.train, signal)
        
        # Process test data  
        X_test, y_test, sid_test = self._prepare_data(self.test, signal)
        
        # Split train into train/val. The sample id of every row is split along (extra array), the
        # split itself (random_state, stratify) is unchanged, so X/y are identical to before.
        X_train, X_val, y_train, y_val, sid_train, sid_val = train_test_split(
            X_train_raw, y_train_raw, sid_train_raw, test_size=0.2, 
            random_state=42, stratify=y_train_raw
        )
        self.sample_names = [signal] + [k for k in self.train if not k.startswith("Signal_")]
        self.sid_train, self.sid_val, self.sid_test = sid_train, sid_val, sid_test
        
        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        self.fitted = (signal, cat, fold)      # the scaler is fitted on the TRAINING half of this fold

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
        
        # index of the sample every row comes from: 0 = signal, i = i-th background (same order as
        # background_data_list), needed to give every process its own training weight
        bkg_names = [k for k in np_features if not k.startswith('Signal_')]
        sid = np.concatenate([np.zeros(len(signal_data), dtype=int)] +
                             [np.full(len(np_features[k]), i + 1, dtype=int) for i, k in enumerate(bkg_names)])
        
        return X, y, sid

    def get_training_data(self, signal: str = "Signal_500", cat="noEFT",
                          scheme: str = TRAIN_SCHEME, min_events: int = TRAIN_MIN_EVENTS,
                          fold: int = 0) -> TrainingData:
        """Arrays + loss weights for the training.

        scheme    : 'baseline' (every event equal) or 'proc:<a>' (process total ~ yield^a), see weights.py
        min_events: background processes with fewer TRAINING events are removed from the train/val sets
                    (0 keeps all). They stay in the test set and in the limit histograms.
        fold      : 0 or 1. Trained on the half that is NOT fold, tested on half `fold` (X_test / y_test / w_test).
        The scaler and the train/val split come from get_signal_background and do not depend on these two
        options, so a model trained with any setting is evaluated with the same scaler later.
        """
        parse_scheme(scheme)                                  # fail early on a typo
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_signal_background(signal, cat, fold)
        names = self.sample_names
        meta = self.output["metadata"]

        n_train = {n: int((self.sid_train == i).sum()) for i, n in enumerate(names) if i > 0}
        phys = {n: physical_event_weight(meta, n) for n in n_train}
        per = process_weights(scheme, n_train, phys, min_events)
        dropped = [n for n, v in per.items() if v == 0.0]

        lut = np.array([1.0] + [per[n] for n in names[1:]])   # index 0 = signal (always kept)
        keep_tr = lut[self.sid_train] > 0
        keep_va = lut[self.sid_val] > 0
        w_train, ref = event_weights(self.sid_train[keep_tr], y_train[keep_tr], per, names)
        w_val, _ = event_weights(self.sid_val[keep_va], y_val[keep_va], per, names, ref)

        # physical weight of every test row (only for the weighted metrics / plots)
        w_lut = np.array([1.0] + [phys[n] for n in names[1:]])
        w_test = w_lut[self.sid_test]

        return TrainingData(
            X_train[keep_tr], y_train[keep_tr], w_train.astype(np.float32),
            X_val[keep_va], y_val[keep_va], w_val.astype(np.float32),
            X_test, y_test, w_test,
            scheme=scheme, min_events=min_events, dropped=dropped, n_train=n_train, process_weight=per,
        )


    def get_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test,
                    w_train=None, w_val=None, seed=TRAIN_SEED):
        """Without w_train / w_val: exactly the old loaders (batches of (X, y), no shuffling).
        With them: train/val batches are (X, y, w) and the training loader shuffles every epoch with a
        generator seeded by `seed` (reproducible). The test loader is always (X, y)."""
        weighted = w_train is not None
        if weighted:
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1),
                                          torch.FloatTensor(w_train).reshape(-1, 1))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1),
                                        torch.FloatTensor(w_val).reshape(-1, 1))
        else:
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).reshape(-1, 1))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val).reshape(-1, 1))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).reshape(-1, 1))

        # Calculate class weights for sampling
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        if weighted:
            gen = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                                      generator=gen, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_test_with_sample(self, signal="Signal_500", cat="noEFT", fold=0):
        """Scaled TEST half (part `fold`) of every sample. fold=0 = the old behaviour."""
        self.train, self.test = self.concatenate_features(signal, cat, fold)
        if self.fitted != (signal, cat, fold):        # scaler state also depends on cat and fold
            self.get_signal_background(signal, cat, fold)
        return {s: self.scaler.transform(x) for s, x in self.test.items()}

    def get_folds_with_sample(self, signal="Signal_500", cat="noEFT"):
        """{sample: [X_part0, X_part1]}: EVERY selected event, scaled, split in the two halves (original order).

        X_part_f is scaled with the scaler of fold f (fitted on the training half of fold f), because it is
        scored by the networks of fold f (trained on the other half). Concatenating part 0 and part 1 gives the
        events in file order, i.e. the order of the weight arrays used in nn_histograms.get_histograms."""
        out = {}
        for fold in (0, 1):
            for s, x in self.get_test_with_sample(signal, cat, fold).items():
                out.setdefault(s, [None, None])[fold] = x
        return out


    def get_total_samples(self, signal="Signal_500", cat="noEFT"):
        self.train, self.test = self.concatenate_features(signal, cat)
        if self.fitted != (signal, cat, 0):
            self.get_signal_background(signal, cat)   # was hard-coded cat="noEFT"
        return {s: self.scaler.transform(np.vstack([self.test[s], self.train[s]]))
                for s in self.test}