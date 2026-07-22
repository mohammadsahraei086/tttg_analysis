import numpy as np
import os
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


from dataclasses import dataclass, field
from typing import Dict, List, Optional

cms_color = {
    "blue": "#3f90da",
    "orange": "#ffa90e",
    "red": "#bd1f01",
    "gray": "#94a4a2",
    "purple": "#832db6",
    "brown": "#a96b59",
    "dark_orange": "#e76300",
    "beige": "#b9ac70",
    "dark_gray": "#717581",
    "light_blue": "#92dadd",
    'cyan': '#17becf',
    'yellow-green': '#bcbd22',
}

@dataclass
class HistInfo:
    ax_label: str
    bins: List = field(default_factory=list)
    backgrounds: Dict[str, List] = field(default_factory=dict)
    data: List = field(default_factory=list)
    signals: Dict[str, List] = field(default_factory=dict)
    # syst: List = field(default_factory=list)

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
    def __init__(self, histograms, normalize=False):
        self.histograms = histograms
        self.normalize = normalize
        self.hist_info = {
            "backgrounds": {},
            "signals": {}
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
                    mc_hist += self.histograms[smpl]
                else:
                    mc_hist = self.histograms[smpl].copy()
            if self.normalize:
                self.hist_info["backgrounds"][bc] = mc_hist.values()/np.sum(mc_hist.values()*np.diff(mc_hist.axes[0].edges))
            else:
                self.hist_info["backgrounds"][bc] = mc_hist.values()

    def _add_signals_info(self, signals):
        for signal in signals:
            hist = self.histograms[signal]
            if self.normalize:
                self.hist_info["signals"][signal] = hist.values()/np.sum(hist.values()*np.diff(hist.axes[0].edges))
            else:
                self.hist_info["signals"][signal] = hist.values()

    def extract_hist_info(self, data=None, MC_BCs=None, signals=None):
        self._add_bins_info()
        # if data is not None:
        #     self._add_data_info(var, cat, data)
        if MC_BCs is not None:
            self._add_MC_BCs_info(MC_BCs)
        if signals is not None:
            self._add_signals_info(signals)

        return HistInfo(**self.hist_info)


class Plotter:
    def __init__(self, hist_info):
        self.hist_info = hist_info
        self.colors = [cms_color["red"], cms_color["orange"], cms_color["purple"],
                       cms_color["beige"], cms_color["blue"], cms_color["dark_gray"],
                       cms_color["light_blue"], cms_color["brown"], cms_color["dark_orange"],
                       cms_color["gray"], cms_color["yellow-green"], cms_color["cyan"]]

    def define_figure(self):
        # self.fig, (self.ax, self.rax) = plt.subplots(
        #     2, 1, figsize=(10, 8), 
        #     gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0}, 
        #     sharex=True
        # )
        self.fig, self.ax = plt.subplots(
            1, figsize=(10, 8)
        )

    def plot_datamc(self):
        
        bottom = np.zeros(len(self.hist_info.bin_centers))
        for i, (name, values) in enumerate(self.hist_info.backgrounds.items()):
            self.ax.bar(
                self.hist_info.bin_centers, values, width=self.hist_info.bin_widths,
                bottom=bottom, alpha=0.8, label=name, color=self.colors[i],
                edgecolor='black', linewidth=0.5
            )
            bottom += values
            
        # self.ax.errorbar(
        #     self.hist_info.bin_centers, self.hist_info.data, yerr=self.hist_info.stat,
        #     fmt='o', color='black', markersize=5, capsize=0,
        #     linewidth=2, label='Data'
        # )
        
        for i, (name, values) in enumerate(self.hist_info.signals.items()):
            self.ax.step(
                self.hist_info.bins, np.append(values, values[-1])*1, where='post',
                alpha=1, linestyle="dashed", label=name, color=self.colors[4], linewidth=2
            )
        
        # lower = self.hist_info.total_backgrounds - self.hist_info.syst
        # upper = self.hist_info.total_backgrounds + self.hist_info.syst
        # self.ax.fill_between(
        #     self.hist_info.bins,
        #     np.append(lower, lower[-1]),
        #     np.append(upper, upper[-1]),
        #     step='post', facecolor="None", alpha=0.9, hatch='////',
        #     label="stat unc.", edgecolor='black', linewidth=0
        # )
        
        # cats = {"emu": "$e\mu$", "ee": "$ee$", "mumu": "$\mu\mu$"}
        # self.ax.text(0.75, 0.45, cats[channel], transform=self.ax.transAxes, 
        #        fontsize=20, fontweight='bold', va='top')

        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=8)
        self.ax.tick_params(axis='both', which='minor', labelsize=12, width=1.5, length=5)
        
        # for label in ax.get_xticklabels() + ax.get_yticklabels():
        #     label.set_fontweight('bold')
        
        for spine in self.ax.spines.values():
            spine.set_linewidth(2)
        
        self.ax.set_ylabel('Events', fontsize=20)
        self.ax.set_xlabel(self.hist_info.ax_label, fontsize=20)
        self.ax.set_xlim(self.hist_info.bins[0], self.hist_info.bins[-1])
        self.ax.legend(fontsize=12, loc='best', frameon=True,
                   framealpha=1.0,
                   edgecolor='black',
                   fancybox=False)
        self.ax.grid(True, alpha=0.3)
        # self.ax.autoscale(enable=True, axis='y', tight=False)
        # self.ax.margins(y=0.2)
        # self.ax.set_yscale('log')
        # self.ax.set_title(f'Normalized differential cross section/{self.x_axis_name}', fontsize=16)
    
    def plot_ratio(self):
        ratio = self.hist_info.data / self.hist_info.total_backgrounds
        ratio_err = self.hist_info.stat / self.hist_info.total_backgrounds

        # Plot ratio
        self.rax.errorbar(self.hist_info.bin_centers, ratio, yerr=ratio_err, 
                          fmt='o', color='black', markersize=5, linewidth=1.5,
                          capsize=0, capthick=1.5, label='Data/MC')

        self.rax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)

        syst_unc_ratio = self.hist_info.syst / self.hist_info.total_backgrounds
        lower = 1 - syst_unc_ratio
        upper = 1 + syst_unc_ratio
        self.rax.fill_between(self.hist_info.bins, np.append(lower, lower[-1]), np.append(upper, upper[-1]),
                 step='post', facecolor="None", alpha=0.9, hatch='////',
                 label='Syst. Unc.', edgecolor='black', linewidth=0)

        # Set labels and limits
        self.rax.set_xlabel(self.hist_info.ax_label, fontsize=15)
        self.rax.set_ylabel('Data/MC')
        self.rax.set_ylim(0.5, 2)
        self.rax.set_xlim(self.hist_info.bins[0], self.hist_info.bins[-1])
        self.rax.grid(True, alpha=0.3)
        self.rax.set_xlim(self.ax.get_xlim())

        # Add bin edges as x-ticks
        # ax_bottom.set_xticks(bins)
    
    def plot_histogram(self, normalized=False):
        self.define_figure()
        self.plot_datamc()
        # self.plot_ratio()

        os.makedirs(f"plots", exist_ok=True)
        if normalized:
            plt.savefig(f"plots/{list(self.hist_info.signals.keys())[0]}_normalized.png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"plots/{list(self.hist_info.signals.keys())[0]}.png", dpi=300, bbox_inches="tight")
        plt.show()
        # plt.savefig(f"plots/{name}.pdf", bbox_inches="tight")
        plt.close()

    