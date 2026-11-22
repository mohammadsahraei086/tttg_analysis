"""Shared loading helpers for the checks in this folder. Read-only: nothing here changes tttg/."""
import numpy as np

from tttg.config import model_path


def load_scores(cat, mass):
    """Trained model -> NN score of every TEST event, per sample.  Returns (dataset, scores)."""
    import torch
    from tttg.data.features import PrepareFeaturesForTraning
    from tttg.training.model import OptimizedHEPClassifier

    signal = f"Signal_{mass}"
    ds = PrepareFeaturesForTraning()
    scaled = ds.get_test_with_sample(signal=signal, cat=cat)     # also fills ds.train / ds.test (raw)
    ncol = next(iter(scaled.values())).shape[1]
    model = OptimizedHEPClassifier(ncol)
    model.load_state_dict(torch.load(model_path(cat, mass), map_location="cpu", weights_only=True))
    model.eval()
    scores = {}
    with torch.no_grad():
        for s, x in scaled.items():
            scores[s] = torch.sigmoid(model(torch.FloatTensor(x))).numpy().ravel()
    return ds, scores


def load_weights(cat, scores):
    """Nominal physical weight (xsec * lumi / sum_genweight, as used for the limit histograms) of every
    TEST event, per sample.
    Background weights are one number per sample, so they are broadcast to one value per event."""
    from tttg.data.weights import WeightManager

    wm = WeightManager()
    out = {}
    for s, sc in scores.items():
        if s.startswith("Signal_"):
            w = wm.get_weights(s, "xsec", "luminosity", "sum_genweight", "variation",
                               cat=cat, variation="Nominal")   # key in WEIGHT_INDEX
        else:
            w = wm.get_weights(s, "xsec", "luminosity", "sum_genweight")
        out[s] = np.broadcast_to(np.asarray(w, dtype=float), (len(sc),)).copy()
    return out