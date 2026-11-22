"""HistInfo container and the extractor that turns per-sample NN histograms into it."""
import math
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List

from tttg.histograms.systematics import compute_systematic_uncertainties


@dataclass
class HistInfo:
    ax_label: str
    bins: List = field(default_factory=list)
    backgrounds: Dict[str, List] = field(default_factory=dict)
    bkg_var: List = field(default_factory=list)
    backgrounds_var: Dict[str, List] = field(default_factory=dict)   # per-group sum of squared weights
    signals_var: Dict[str, List] = field(default_factory=dict)       # per-signal sum of squared weights (MC stat)
    data: List = field(default_factory=list)
    signals: Dict[str, List] = field(default_factory=dict)
    syst: dict[str, dict[str, list]] = field(default_factory=dict)

    @property
    def stat(self) -> List[float]:
        """Compute statistical errors dynamically."""
        return [math.sqrt(abs(x)) for x in self.data]

    # @property
    # def syst(self) -> List[float]:
    #     """Compute syst errors dynamically."""
    #     return [math.sqrt(abs(x)) for x in self.total_backgrounds]

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
    def __init__(self, histograms, normalize=False):
        self.histograms = histograms
        self.normalize = normalize
        self.hist_info = {
            "backgrounds": {},
            "backgrounds_var": {},
            "signals": {},
            "signals_var": {},
            "syst": {},
            "bkg_var": 0.0
        }

    def _add_bins_info(self):
        smpl = list(self.histograms.keys())[0]
        hist = self.histograms[smpl]
        self.hist_info["bins"] = hist.axes[0].edges
        self.hist_info["ax_label"] = "NN_output"

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

    def _add_MC_BCs_info(self, MC_BCs):
        for bc, bc_lst in MC_BCs.items():
            mc_hist = None
            for smpl in bc_lst:
                if mc_hist is not None:
                    mc_hist += self.histograms[smpl][{"variation": "Nominal"}]
                else:
                    mc_hist = self.histograms[smpl][{"variation": "Nominal"}].copy()
            self.hist_info["bkg_var"] = self.hist_info["bkg_var"] + mc_hist.variances()
            self.hist_info["backgrounds_var"][bc] = mc_hist.variances()
            if self.normalize:
                self.hist_info["backgrounds"][bc] = mc_hist.values()/np.sum(mc_hist.values()*np.diff(mc_hist.axes[0].edges))
            else:
                self.hist_info["backgrounds"][bc] = mc_hist.values()

    def _add_signals_info(self, signals):
        for signal in signals:
            hist = self.histograms[signal][{"variation": "Nominal"}]
            self.hist_info["signals_var"][signal] = hist.variances()
            if self.normalize:
                self.hist_info["signals"][signal] = hist.values()/np.sum(hist.values()*np.diff(hist.axes[0].edges))
            else:
                self.hist_info["signals"][signal] = hist.values()

    def add_syst_info(self, signals: list) -> None:
        for signal in signals:
            self.hist_info["syst"][signal] = compute_systematic_uncertainties(signal, self.histograms)
        

    def extract_hist_info(self, data=None, MC_BCs=None, signals=None, syst=True):
        self._add_bins_info()
        # if data is not None:
        #     self._add_data_info(var, cat, data)
        if MC_BCs is not None:
            self._add_MC_BCs_info(MC_BCs)
        if signals is not None:
            self._add_signals_info(signals)
        if syst:
            self.add_syst_info(signals)

        return HistInfo(**self.hist_info)
