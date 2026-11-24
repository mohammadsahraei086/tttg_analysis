import numpy as np
from coffea.util import load

from tttg.config import COFFEA_OUTPUT, LUMINOSITY

#########################################################################
######     Variation     ########
#########################################################################

NOMINAL_VARIATION = "MUF=1.0_MUR=1.0_PDF=247000"

WEIGHT_INDEX = {
    # 'MUF=0.5_MUR=0.5_PDF=247000': 181,
    'MUF=0.5_MUR=1.0_PDF=247000': 15,
    # 'MUF=0.5_MUR=2.0_PDF=247000': 183,
    'MUF=1.0_MUR=0.5_PDF=247000': 5,
    'Nominal': 44,
    'MUF=1.0_MUR=1.0_PDF=247001': 45,
    'MUF=1.0_MUR=1.0_PDF=247002': 46,
    'MUF=1.0_MUR=1.0_PDF=247003': 47,
    'MUF=1.0_MUR=1.0_PDF=247004': 48,
    'MUF=1.0_MUR=1.0_PDF=247005': 49,
    'MUF=1.0_MUR=1.0_PDF=247006': 50,
    'MUF=1.0_MUR=1.0_PDF=247007': 51,
    'MUF=1.0_MUR=1.0_PDF=247008': 52,
    'MUF=1.0_MUR=1.0_PDF=247009': 53,
    'MUF=1.0_MUR=1.0_PDF=247010': 54,
    'MUF=1.0_MUR=1.0_PDF=247011': 55,
    'MUF=1.0_MUR=1.0_PDF=247012': 56,
    'MUF=1.0_MUR=1.0_PDF=247013': 57,
    'MUF=1.0_MUR=1.0_PDF=247014': 58,
    'MUF=1.0_MUR=1.0_PDF=247015': 59,
    'MUF=1.0_MUR=1.0_PDF=247016': 60,
    'MUF=1.0_MUR=1.0_PDF=247017': 61,
    'MUF=1.0_MUR=1.0_PDF=247018': 62,
    'MUF=1.0_MUR=1.0_PDF=247019': 63,
    'MUF=1.0_MUR=1.0_PDF=247020': 64,
    'MUF=1.0_MUR=1.0_PDF=247021': 65,
    'MUF=1.0_MUR=1.0_PDF=247022': 66,
    'MUF=1.0_MUR=1.0_PDF=247023': 67,
    'MUF=1.0_MUR=1.0_PDF=247024': 68,
    'MUF=1.0_MUR=1.0_PDF=247025': 69,
    'MUF=1.0_MUR=1.0_PDF=247026': 70,
    'MUF=1.0_MUR=1.0_PDF=247027': 71,
    'MUF=1.0_MUR=1.0_PDF=247028': 72,
    'MUF=1.0_MUR=1.0_PDF=247029': 73,
    'MUF=1.0_MUR=1.0_PDF=247030': 74,
    'MUF=1.0_MUR=1.0_PDF=247031': 75,
    'MUF=1.0_MUR=1.0_PDF=247032': 76,
    'MUF=1.0_MUR=1.0_PDF=247033': 77,
    'MUF=1.0_MUR=1.0_PDF=247034': 78,
    'MUF=1.0_MUR=1.0_PDF=247035': 79,
    'MUF=1.0_MUR=1.0_PDF=247036': 80,
    'MUF=1.0_MUR=1.0_PDF=247037': 81,
    'MUF=1.0_MUR=1.0_PDF=247038': 82,
    'MUF=1.0_MUR=1.0_PDF=247039': 83,
    'MUF=1.0_MUR=1.0_PDF=247040': 84,
    'MUF=1.0_MUR=1.0_PDF=247041': 85,
    'MUF=1.0_MUR=1.0_PDF=247042': 86,
    'MUF=1.0_MUR=1.0_PDF=247043': 87,
    'MUF=1.0_MUR=1.0_PDF=247044': 88,
    'MUF=1.0_MUR=1.0_PDF=247045': 89,
    'MUF=1.0_MUR=1.0_PDF=247046': 90,
    'MUF=1.0_MUR=1.0_PDF=247047': 91,
    'MUF=1.0_MUR=1.0_PDF=247048': 92,
    'MUF=1.0_MUR=1.0_PDF=247049': 93,
    'MUF=1.0_MUR=1.0_PDF=247050': 94,
    'MUF=1.0_MUR=1.0_PDF=247051': 95,
    'MUF=1.0_MUR=1.0_PDF=247052': 96,
    'MUF=1.0_MUR=1.0_PDF=247053': 97,
    'MUF=1.0_MUR=1.0_PDF=247054': 98,
    'MUF=1.0_MUR=1.0_PDF=247055': 99,
    'MUF=1.0_MUR=1.0_PDF=247056': 100,
    'MUF=1.0_MUR=1.0_PDF=247057': 101,
    'MUF=1.0_MUR=1.0_PDF=247058': 102,
    'MUF=1.0_MUR=1.0_PDF=247059': 103,
    'MUF=1.0_MUR=1.0_PDF=247060': 104,
    'MUF=1.0_MUR=1.0_PDF=247061': 105,
    'MUF=1.0_MUR=1.0_PDF=247062': 106,
    'MUF=1.0_MUR=1.0_PDF=247063': 107,
    'MUF=1.0_MUR=1.0_PDF=247064': 108,
    'MUF=1.0_MUR=1.0_PDF=247065': 109,
    'MUF=1.0_MUR=1.0_PDF=247066': 110,
    'MUF=1.0_MUR=1.0_PDF=247067': 111,
    'MUF=1.0_MUR=1.0_PDF=247068': 112,
    'MUF=1.0_MUR=1.0_PDF=247069': 113,
    'MUF=1.0_MUR=1.0_PDF=247070': 114,
    'MUF=1.0_MUR=1.0_PDF=247071': 115,
    'MUF=1.0_MUR=1.0_PDF=247072': 116,
    'MUF=1.0_MUR=1.0_PDF=247073': 117,
    'MUF=1.0_MUR=1.0_PDF=247074': 118,
    'MUF=1.0_MUR=1.0_PDF=247075': 119,
    'MUF=1.0_MUR=1.0_PDF=247076': 120,
    'MUF=1.0_MUR=1.0_PDF=247077': 121,
    'MUF=1.0_MUR=1.0_PDF=247078': 122,
    'MUF=1.0_MUR=1.0_PDF=247079': 123,
    'MUF=1.0_MUR=1.0_PDF=247080': 124,
    'MUF=1.0_MUR=1.0_PDF=247081': 125,
    'MUF=1.0_MUR=1.0_PDF=247082': 126,
    'MUF=1.0_MUR=1.0_PDF=247083': 127,
    'MUF=1.0_MUR=1.0_PDF=247084': 128,
    'MUF=1.0_MUR=1.0_PDF=247085': 129,
    'MUF=1.0_MUR=1.0_PDF=247086': 130,
    'MUF=1.0_MUR=1.0_PDF=247087': 131,
    'MUF=1.0_MUR=1.0_PDF=247088': 132,
    'MUF=1.0_MUR=1.0_PDF=247089': 133,
    'MUF=1.0_MUR=1.0_PDF=247090': 134,
    'MUF=1.0_MUR=1.0_PDF=247091': 135,
    'MUF=1.0_MUR=1.0_PDF=247092': 136,
    'MUF=1.0_MUR=1.0_PDF=247093': 137,
    'MUF=1.0_MUR=1.0_PDF=247094': 138,
    'MUF=1.0_MUR=1.0_PDF=247095': 139,
    'MUF=1.0_MUR=1.0_PDF=247096': 140,
    'MUF=1.0_MUR=1.0_PDF=247097': 141,
    'MUF=1.0_MUR=1.0_PDF=247098': 142,
    'MUF=1.0_MUR=1.0_PDF=247099': 143,
    'MUF=1.0_MUR=1.0_PDF=247100': 144,
    'MUF=1.0_MUR=2.0_PDF=247000': 34,
    # 'MUF=2.0_MUR=0.5_PDF=247000': 287,
    'MUF=2.0_MUR=1.0_PDF=247000': 24,
    # 'MUF=2.0_MUR=2.0_PDF=247000': 289,
    "Nominal_isrMuR_fsrMuR": 0,
    "fsrMuRDown": 152,
    "fsrMuRUp": 153,
    "isrMuRDown": 150,
    "isrMuRUp": 151,
}

# Convenience: ordered list of variation names, ready to loop over when
# filling histograms. Includes the central PDF+scale entry (index 185),
# which is a useful cross-check since its ratio to itself is always 1.
VARIATIONS = list(WEIGHT_INDEX.keys())

###########################################################################
##############     Weights     ################################
###########################################################################

class WeightManager:
    def __init__(self):
        self.output = load(str(COFFEA_OUTPUT))
        self.phase2_luminosity = LUMINOSITY
    
    def get_weights(self, sample, *weights, **kwargs):
        total_weight = np.prod([])
        for weight in weights:
            new_weight = getattr(self, weight)
            total_weight = total_weight * new_weight(sample, **kwargs)
        return total_weight
    
    def xsec(self, sample, **kwargs):
        return self.output["metadata"][sample]["xsec"]
    
    def luminosity(self, sample, **kwargs):
        return self.phase2_luminosity
    
    def sum_genweight(self, sample, use="test", **kwargs):
        """1/N_gen per selected event, times 2 if only one half of the selected events is used.
        use="test": one half only (old behaviour, factor 2). use="all": every selected event (2-fold
        cross-training, nn_histograms.py) -> no factor 2."""
        factor = {"test": 2.0, "all": 1.0}[use]
        return factor/self.output["metadata"][sample]["nevents"]

    def variation(self, sample, cat="noEFT", variation=NOMINAL_VARIATION, use="test", **kwargs):
        """Per-event scale/PDF systematic weight, expressed as a ratio to
        the nominal (MUF=1, MUR=1, PDF=247000) weight.

        Multiplying this ratio into the usual xsec*luminosity/sum_genweight
        normalization reweights both the *shape* and the *rate* of the
        distribution for the requested variation: since the LHE sample is
        unweighted with event_norm=average, average(w_variation)/average(w_nominal)
        over the full sample equals xsec_variation/xsec_nominal, so no
        separate per-variation cross section needs to be looked up.

        `variation` must be one of the keys in WEIGHT_INDEX (this module).
        Defaults to the nominal weight itself (ratio == 1) if not given.
        use="test": only the first half of the events (old behaviour); use="all": every event, in the original
        order (the two halves = fold 0 and fold 1 of the 2-fold training, see features.py).
        """
        if "srMuR" in variation:
            w = self.output["variations"][cat][sample]["Weight"].value
            nominal_idx = WEIGHT_INDEX["Nominal_isrMuR_fsrMuR"]
        else:
            w = self.output["variations"][cat][sample]["LHEF"].value
            nominal_idx = WEIGHT_INDEX["Nominal"]
        if use == "test":
            w = w[:int(len(w)/2)]
        elif use != "all":
            raise ValueError(f"use must be 'test' or 'all', got {use!r}")
        idx = WEIGHT_INDEX[variation]
        return w[:, idx] / w[:, nominal_idx]

###########################################################################
##############     Training weights (NN loss)     #########################
###########################################################################
# Added to let the NN training see the same background composition as the limit histograms.
# Pure functions (no coffea access) so that they are easy to test; features.py calls them.

def parse_scheme(scheme: str):
    """'baseline' -> None, 'proc:<a>' -> a (float). Anything else is an error."""
    if scheme == "baseline":
        return None
    if isinstance(scheme, str) and scheme.startswith("proc:"):
        try:
            alpha = float(scheme.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"bad scheme {scheme!r}: the exponent in 'proc:<a>' must be a number")
        if alpha < 0:
            raise ValueError(f"bad scheme {scheme!r}: the exponent must be >= 0")
        return alpha
    raise ValueError(f"unknown scheme {scheme!r} (use 'baseline' or 'proc:<a>', e.g. 'proc:0.5')")


def physical_event_weight(metadata, sample, luminosity=LUMINOSITY):
    """Nominal weight of ONE event of a background sample, xsec * lumi * 2 / N_gen.

    Same number as WeightManager.get_weights(sample, 'xsec', 'luminosity', 'sum_genweight') that fills the
    limit histograms (the factor 2 is there because only half of the events, the test half, is used there).
    """
    m = metadata[sample]
    return float(m["xsec"]) * luminosity * 2.0 / m["nevents"]


def process_weights(scheme, n_train, phys_w, min_events=0):
    """Per-EVENT training weight of every background process.

    n_train  : {sample: number of training events}
    phys_w   : {sample: physical weight of one event}
    min_events: processes with fewer training events get weight 0 (= left out of the training)
    The process as a whole gets (yield)^alpha, yield = phys_w * n_train, spread uniformly over its events.
    'baseline' gives 1.0 to everything that is kept.
    """
    alpha = parse_scheme(scheme)
    out = {}
    for s, n in n_train.items():
        if n < min_events or n == 0:
            out[s] = 0.0
        elif alpha is None:
            out[s] = 1.0
        else:
            out[s] = (phys_w[s] * n) ** alpha / n
    return out


def event_weights(sid, y, per_sample, names, ref=None):
    """Per-event weights for the rows of one set (train or val).

    sid       : sample index (into `names`) of every row;  y: 1 = signal, 0 = background
    per_sample: {background sample: per-event weight} from process_weights
    The signal rows get one common weight such that signal and background have the same total weight
    (class balance, replaces BCE's pos_weight), then everything is divided by the mean weight of the
    TRAINING set so that the loss keeps its usual size. For the validation set pass ref=<value returned for
    the training set>, so that both sets use the same factors.
    Returns (weights, ref).
    """
    w = np.ones(len(y), dtype=np.float64)
    bkg = y == 0
    lut = np.array([per_sample.get(n, 1.0) for n in names])
    w[bkg] = lut[sid[bkg]]
    if ref is None:
        if w[bkg].sum() <= 0:
            raise ValueError("all background weights are zero (lower min_events)")
        sig_scale = w[bkg].sum() / (~bkg).sum()
        w[~bkg] = sig_scale
        norm = w.mean()
        ref = (sig_scale, norm)
    else:
        w[~bkg] = ref[0]
    return w / ref[1], ref
