import numpy as np
import math

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class HistInfo:
    var_name: str
    ax_label: str
    cat: str
    bins: List = field(default_factory=list)
    backgrounds: Dict[str, List] = field(default_factory=dict)
    data: List = field(default_factory=list)
    signals: Dict[str, List] = field(default_factory=dict)
    # syst: List = field(default_factory=list)
    has_plot: bool = False 

    @property
    def stat(self) -> List[float]:
        """Compute statistical errors dynamically."""
        return [math.sqrt(abs(x)) for x in self.data]

    # Temporary syst
    @property
    def syst(self) -> List[float]:
        """Compute syst errors dynamically."""
        return [math.sqrt(abs(x)) for x in self.total_backgrounds]

    @property
    def total_backgrounds(self) -> List[float]:
        return sum(list(self.backgrounds.values()))

    @property
    def bin_centers(self) -> List:
        centers = []
        for i in range(len(self.bins)-1):
            center = (self.bins[i] + self.bins[i+1]) / 2
            centers.append(center)
        return centers

    @property
    def bin_widths(self) -> List:
        return np.diff(self.bins)


class ExtractHistData:
    """
    This class handles coffea output reading and creating HistInfo for HistogramPlotter
    """
    def __init__(self, coffea_output, normalize=False):
        self.output = coffea_output
        self.normalize = normalize
        self.hist_info = {
            "backgrounds": {},
            "signals": {},
            "has_plot": False
        }

    def _add_bins_info(self, cat:str, var:str):
        smpl = list(self.output["hists"][cat][var].keys())[0]
        hist = self.output["hists"][cat][var][smpl]
        self.hist_info["bins"] = hist.axes[0].edges
        self.hist_info["ax_label"] = hist.axes[0].label

    # def _add_data_info(self, var, cat, data):
    #     data_hist = None
    #     for smpl in data[self.year]:
    #         for hist in self.output["hists"][var][smpl].values():
    #             if data_hist is not None:
    #                 data_hist += hist
    #             else:
    #                 data_hist = hist
    #     if cat in data_hist.axes[0]:
    #         self.hist_info["data"] = data_hist[{"cat": cat}].values()
    #         self.hist_info["has_plot"] = True

    def _add_MC_BCs_info(self, var, cat, MC_BCs):
        for bc, bc_lst in MC_BCs.items():
            mc_hist = None
            for smpl in bc_lst:
                if mc_hist is not None:
                    mc_hist += self.output["hists"][cat][var][smpl]
                else:
                    mc_hist = self.output["hists"][cat][var][smpl]
            if self.normalize:
                self.hist_info["backgrounds"][bc] = mc_hist.values()/np.sum(mc_hist.values()*np.diff(mc_hist.axes[0].edges))
            else:
                self.hist_info["backgrounds"][bc] = mc_hist.values()
            self.hist_info["has_plot"] = True

    def _add_signals_info(self, var, cat, signals):
        for signal in signals:
            hist = self.output["hists"][cat][var][signal]
            if self.normalize:
                self.hist_info["signals"][signal] = hist.values()/np.sum(hist.values()*np.diff(hist.axes[0].edges))
            else:
                self.hist_info["signals"][signal] = hist.values()
            self.hist_info["has_plot"] = True

    def extract_hist_info(self, var, cat, data=None, MC_BCs=None, signals=None):
        self.hist_info["var_name"] = var
        self.hist_info["cat"] = cat
        self._add_bins_info(cat, var)
        if data is not None:
            self._add_data_info(var, cat, data)
        if MC_BCs is not None:
            self._add_MC_BCs_info(var, cat, MC_BCs)
        if signals is not None:
            self._add_signals_info(var, cat, signals)

        return HistInfo(**self.hist_info)
    