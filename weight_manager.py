import numpy as np

class WeightManager:
    def __init__(self, n_primary):
        self.phase2_luminosity = 3000
        self.n_primary = n_primary
    
    def get_weights(self, events, *weights, **kwargs):
        total_weight = np.prod([])
        for weight in weights:
            new_weight = getattr(self, weight)
            total_weight = total_weight * new_weight(events, **kwargs)
        return total_weight
    
    def xsec(self, events):
        return events.metadata["xsec"]*1000
    
    def luminosity(self, events):
        return self.phase2_luminosity
    
    def sum_genweight(self, events):
        return 1./self.n_primary