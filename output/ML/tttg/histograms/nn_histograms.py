"""Evaluate a trained NN on all samples and fill weighted NN-output histograms."""
import numpy as np
import hist

import torch

from tttg.config import NN_BINS, model_paths
from tttg.data.weights import WeightManager, VARIATIONS
from tttg.training.model import OptimizedHEPClassifier


def get_histograms(
    m: int,
    test_with_sample: dict[str, np.ndarray],
    cat: str="noEFT"
) -> dict[str, hist.Hist]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = test_with_sample[list(test_with_sample.keys())[0]].shape[1]
    
    # NN ensemble: every member (one per seed) scores each event and the sigmoid outputs are averaged
    paths = model_paths(cat, m)
    if not paths:
        raise FileNotFoundError(f"no trained model for {cat} Signal_{m} (run tttg.training.train_eval)")
    models = []
    for path in paths:
        model = OptimizedHEPClassifier(input_dim=input_dim)
        model.load_state_dict(torch.load(path, weights_only=True))
        models.append(model.eval().to(device))
    print(f"{cat} Signal_{m}: NN ensemble of {len(models)} model(s)")

    histograms = {}
    
    weight_manager = WeightManager() 
    for smpl in test_with_sample.keys():

        # if smpl == "WJets" or smpl == "ZJets":
        #     continue    
        test_tensor = torch.FloatTensor(test_with_sample[smpl]).to(device)
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
        
        with torch.no_grad():
            sigma_output = torch.stack(
                [torch.sigmoid(mdl(test_tensor)).flatten() for mdl in models]
            ).mean(dim=0).cpu().numpy()

        if "Signal_" in smpl:
            for variation_name in VARIATIONS:
                varied_weight = weight_manager.get_weights(
                    smpl, "xsec", "luminosity", "sum_genweight", "variation", cat=cat, variation=variation_name
                )
                if len(varied_weight) != len(sigma_output):
                    raise ValueError(
                        f"{cat}/{smpl}/{variation_name}: {len(varied_weight)} weights "
                        f"vs {len(sigma_output)} events"
                    )
                h.fill(sigma_output, variation=variation_name, weight=varied_weight)
        else:
            weight = weight_manager.get_weights(
                smpl, "xsec", "luminosity", "sum_genweight"
            )
            h.fill(sigma_output, variation="Nominal", weight=weight)

        histograms[smpl] = h.copy()
        h.reset()

    return histograms