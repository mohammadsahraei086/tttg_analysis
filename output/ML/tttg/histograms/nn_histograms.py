"""Evaluate a trained NN on all samples and fill weighted NN-output histograms."""
import numpy as np
import hist

import torch

from tttg.config import NN_BINS, model_paths
from tttg.data.weights import WeightManager, VARIATIONS
from tttg.training.model import OptimizedHEPClassifier


def _load_ensemble(cat, m, fold, input_dim, device):
    """The networks of one fold (one per seed), in eval mode."""
    paths = model_paths(cat, m, fold=fold)
    if not paths:
        raise FileNotFoundError(f"no trained model for {cat} Signal_{m} fold {fold} (run tttg.training.train_eval)")
    models = []
    for path in paths:
        model = OptimizedHEPClassifier(input_dim=input_dim)
        model.load_state_dict(torch.load(path, weights_only=True))
        models.append(model.eval().to(device))
    print(f"{cat} Signal_{m} fold {fold}: NN ensemble of {len(models)} model(s)")
    return models


def get_histograms(
    m: int,
    folds_with_sample: dict[str, list[np.ndarray]],
    cat: str="noEFT"
) -> dict[str, hist.Hist]:
    """NN-output histograms of ALL selected events (2-fold cross-training).

    folds_with_sample: {sample: [X_part0, X_part1]} from PrepareFeaturesForTraning.get_folds_with_sample.
    Every event is scored ONLY by the networks that did not see it in training: part f by the ensemble of fold f
    (trained on the other half). The scores are put back in the original event order, so every event is filled
    with its own weight xsec * lumi / N_gen (no factor 2: all selected events are used)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first = folds_with_sample[next(iter(folds_with_sample))]
    if isinstance(first, np.ndarray):
        raise TypeError("get_histograms now needs {sample: [X_part0, X_part1]} -> use "
                        "PrepareFeaturesForTraning.get_folds_with_sample instead of get_test_with_sample")
    input_dim = first[0].shape[1]

    # NN ensemble per fold: every member (one per seed) scores each event and the sigmoid outputs are averaged
    ensembles = [_load_ensemble(cat, m, f, input_dim, device) for f in (0, 1)]

    histograms = {}
    
    weight_manager = WeightManager() 
    for smpl, parts in folds_with_sample.items():

        # if smpl == "WJets" or smpl == "ZJets":
        #     continue    
        axes = []

        axes.append(hist.axis.Regular(bins=NN_BINS, start=0, stop=1, name="NN_response", label="NN_output"))
        axes.append(
            hist.axis.StrCategory(
                    VARIATIONS,
                    name="variation",
                    label="Scale/PDF systematic variation",
                    growth=False,
                )
        )
            
        h = hist.Hist(*axes, storage=hist.storage.Weight(), name="NN_output", label="NN_output")
        
        scores = []
        with torch.no_grad():
            for fold, X in enumerate(parts):          # part f is scored by the networks of fold f
                x_t = torch.FloatTensor(X).to(device)
                scores.append(torch.stack(
                    [torch.sigmoid(mdl(x_t)).flatten() for mdl in ensembles[fold]]
                ).mean(dim=0).cpu().numpy())
        sigma_output = np.concatenate(scores)         # part 0 then part 1 = original event order

        if "Signal_" in smpl:
            for variation_name in VARIATIONS:
                varied_weight = weight_manager.get_weights(
                    smpl, "xsec", "luminosity", "sum_genweight", "variation",
                    cat=cat, variation=variation_name, use="all"
                )
                if len(varied_weight) != len(sigma_output):
                    raise ValueError(
                        f"{cat}/{smpl}/{variation_name}: {len(varied_weight)} weights "
                        f"vs {len(sigma_output)} events"
                    )
                h.fill(sigma_output, variation=variation_name, weight=varied_weight)
        else:
            weight = weight_manager.get_weights(
                smpl, "xsec", "luminosity", "sum_genweight", use="all"
            )
            h.fill(sigma_output, variation="Nominal", weight=weight)

        histograms[smpl] = h.copy()
        h.reset()

    return histograms
