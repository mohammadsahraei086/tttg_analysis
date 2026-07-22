import numpy as np
import hist
import awkward as ak
from dataclasses import dataclass
from typing import Callable, List, Optional

from axis_selection import *

from weight_manager import WeightManager

@dataclass
class Axis:
    name: str 
    label: str 
    bins: int = None
    start: float = None
    stop: float = None
    type: str = "regular"
    function: Optional[Callable] = None
    growth = True
    
    def __post_init__(self):
        if self.function is not None:
            self.get_variable = self.function
    
    def get_variable(self, events):
        raise NotImplementedError(f"Provide a function parameter when creating Axis {self.name}")

class Histogram:
    def __init__(self,name, axes:List[Axis], weights=None):
        self.name = name
        self.axes = axes
        hist_axis = []
        for axis in self.axes:
            hist_axis.append(self.get_hist_axis(axis))
        self.weights = weights
        self.histogram = hist.Hist(*hist_axis, name=name, storage="weight")

    def get_hist_axis(self, ax: Axis):
        if ax.name == None:
            ax.name = f"{ax.coll}.{ax.field}"
        if ax.type == "regular" and isinstance(ax.bins, list):
            ax.type = "variable"
        if ax.type == "regular":
            return hist.axis.Regular(
                name=ax.name,
                bins=ax.bins,
                start=ax.start,
                stop=ax.stop,
                label=ax.label,
            )
        elif ax.type == "variable":
            if not isinstance(ax.bins, list):
                raise ValueError(
                    "A list of bins edges is needed as 'bins' parameters for a type='variable' axis"
                )
            return hist.axis.Variable(
                ax.bins,
                name=ax.name,
                label=ax.label
            )
        elif ax.type == "int":
            return hist.axis.Integer(
                name=ax.name,
                start=ax.start,
                stop=ax.stop,
                label=ax.label
            )
        elif ax.type == "intcat":
            return hist.axis.IntCategory(
                ax.bins,
                name=ax.name,
                label=ax.label
            )
        elif ax.type == "strcat":
            return hist.axis.StrCategory(
                ax.bins, name=ax.name, label=ax.label, growth=ax.growth
            )

    def fill(self, events, n_primary):
        ax = {}
        for axis in self.axes:
            ax[axis.name] = axis.get_variable(events)
        if self.weights is not None:
            weight_manager = WeightManager(n_primary)
            weight = weight_manager.get_weights(events, *self.weights)
            self.histogram.fill(**ax, weight=weight)
        else:
            self.histogram.fill(**ax)

    def get_histogram(self):
        return self.histogram
    
    def reset_histogram(self):
        self.histogram.reset()

class HistManager:
    def __init__(self):
        self.axes = {}
        self.histograms = {}
        
    def define_axes(self):
        # self.add_axis("electron_pt",
        #               "$p_T^e (GeV)$",
        #               [ 20.,  35.,  50.,  70., 100., 130., 165., 200., 250., 300.],
        #               function = lambda events: ak.to_numpy(ak.flatten(events.GoodElectrons.PT)))
        # self.add_axis("muon_pt",
        #               "$p_T^{\mu} (GeV)$",
        #               [ 20.,  35.,  50.,  70., 100., 130., 165., 200., 250., 300.],
        #               function = lambda events: ak.flatten(events.GoodMuons.PT))
        self.add_axis("lepton_pt",
                      "$p_T^{\ell} (GeV)$",
                      bins=15,
                      start=20,
                      stop=320,
                      function = lambda events: ak.flatten(events.GoodLeptons.PT))
        self.add_axis("lepton_eta",
                      "$\eta^e$",
                      bins=12,
                      start=-3,
                      stop=3,
                      function = lambda events: ak.to_numpy(ak.flatten(events.GoodLeptons.eta)))
        self.add_axis("jet_pt",
                      "$p_T^{jets} (GeV)$",
                      bins=15,
                      start=25,
                      stop=325,
                      function = lambda events: ak.flatten(events.GoodJets.PT))
        self.add_axis("jet_eta",
                      "$\eta^{jets}$",
                      bins=20,
                      start=-5,
                      stop=5,
                      function = lambda events: ak.flatten(events.GoodJets.eta))
        self.add_axis("bjet_pt",
                      "$p_T^{b-jets} (GeV)$",
                      bins=15,
                      start=25,
                      stop=325,
                      function = lambda events: ak.flatten(events.GoodBJets.PT))
        self.add_axis("bjet_eta",
                      "$\eta^{b-jets}$",
                      bins=12,
                      start=-3,
                      stop=3,
                      function = lambda events: ak.flatten(events.GoodBJets.eta))
        self.add_axis("met_pt",
                      "$p_T^{MET} (GeV)$",
                      bins=20,
                      start=25,
                      stop=425,
                      function = lambda events: events.MissingET.MET)
        self.add_axis("met_eta",
                      "$\eta^{MET} (GeV)$",
                      bins=12,
                      start=-3,
                      stop=3,
                      function = lambda events: events.MissingET.eta)
        self.add_axis("ht_jets",
                      "$H_T^{Jets} (GeV)$",
                      bins=20,
                      start=0,
                      stop=2000,
                      function = lambda events: events.HT_Jets)
        self.add_axis("ht_goodJets",
                      "$H_T^{Good_Jets} (GeV)$",
                      bins=20,
                      start=600,
                      stop=2100,
                      function = lambda events: events.HT_GoodJets)
        self.add_axis("m_tw",
                      "$M_T^W (GeV)$",
                      bins=20,
                      start=0,
                      stop=300,
                      function = lambda events: ak.flatten(events.W_T))
        self.add_axis("m_w",
                      "$M_W (GeV)$",
                      bins=20,
                      start=0,
                      stop=200,
                      function = lambda events: ak.flatten(events.W.mass))
        self.add_axis("w_pt",
                      "$p_T^W (GeV)$",
                      bins=20,
                      start=0,
                      stop=300,
                      function = lambda events: ak.flatten(events.W.pt))
        self.add_axis("w_eta",
                      "$\eta^W (GeV)$",
                      bins=12,
                      start=-3,
                      stop=3,
                      function = lambda events: ak.flatten(events.W.eta))
        self.add_axis("m_top",
                      "$M_{top} (GeV)$",
                      bins=18,
                      start=100,
                      stop=1000,
                      function = lambda events: ak.flatten(events.top.mass))
        self.add_axis("top_pt",
                      "$p_T^{top} (GeV)$",
                      bins=14,
                      start=0,
                      stop=700,
                      function = lambda events: ak.flatten(events.top.pt))
        self.add_axis("top_eta",
                      "$\eta^top(GeV)$",
                      bins=12,
                      start=-3,
                      stop=3,
                      function = lambda events: ak.flatten(events.top.eta))
        self.add_axis("m_t",
                      "$M_{T} (GeV)$",
                      bins=18,
                      start=200,
                      stop=2000,
                      function = lambda events: ak.flatten(events.T.mass))
        self.add_axis("t_pt",
                      "$p_T^{T} (GeV)$",
                      bins=14,
                      start=0,
                      stop=700,
                      function = lambda events: ak.flatten(events.T.pt))
        self.add_axis("t_eta",
                      "$\eta^T (GeV)$",
                      bins=12,
                      start=-3,
                      stop=3,
                      function = lambda events: ak.flatten(events.T.eta))

        #############################     delta r       #################
        self.add_axis("delta_r_ljet",
                      "$\Delta r(\ell,Leading-Jet)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.GoodLeptons.delta_r(events.GoodNotBJets[:,0])))
        self.add_axis("delta_r_lbjet",
                      "$\Delta r(\ell,Leading-bJet)$",
                      bins=13,
                      start=0.4,
                      stop=4.4,
                      function = lambda events: ak.flatten(events.GoodLeptons.delta_r(events.GoodBJets[:,0])))
        self.add_axis("delta_r_wl",
                      "$\Delta r(W, \ell)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.W.delta_r(events.GoodLeptons[:,0])))
        self.add_axis("delta_r_wjet",
                      "$\Delta r(W, Leading-Jet)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.W.delta_r(events.GoodNotBJets[:,0])))
        self.add_axis("delta_r_wbjet",
                      "$\Delta r(W, Leading-bJet)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.W.delta_r(events.GoodBJets[:,0])))
        self.add_axis("delta_r_topl",
                      "$\Delta r(top, \ell)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.top.delta_r(events.GoodLeptons[:,0])))
        self.add_axis("delta_r_topjet",
                      "$\Delta r(top, Leading-Jet)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.top.delta_r(events.GoodNotBJets[:,0])))
        self.add_axis("delta_r_topbjet",
                      "$\Delta r(top, Leading-bJet)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.top.delta_r(events.GoodBJets[:,0])))
        self.add_axis("delta_r_tl",
                      "$\Delta r(T, \ell)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.T.delta_r(events.GoodLeptons[:,0])))
        self.add_axis("delta_r_tjet",
                      "$\Delta r(T, Leading-Jet)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.T.delta_r(events.GoodNotBJets[:,0])))
        self.add_axis("delta_r_tbjet",
                      "$\Delta r(T, Leading-bJet)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.T.delta_r(events.GoodBJets[:,0])))
        self.add_axis("delta_r_wtop",
                      "$\Delta r(W, top)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.W.delta_r(events.top[:,0])))
        self.add_axis("delta_r_wt",
                      "$\Delta r(W, t)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.W.delta_r(events.T[:,0])))
        self.add_axis("delta_r_ttop",
                      "$\Delta r(t, top)$",
                      bins=15,
                      start=0.4,
                      stop=5,
                      function = lambda events: ak.flatten(events.T.delta_r(events.top[:,0])))

        ################################    delta phi ###################
        self.add_axis("delta_phi_ljet",
                      "$Cos(\Delta \phi(\ell,Leading-Jet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.GoodLeptons.delta_phi(events.GoodNotBJets[:,0])))))
        self.add_axis("delta_phi_lbjet",
                      "$Cos(\Delta \phi(\ell,Leading-bJet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.GoodLeptons.delta_phi(events.GoodBJets[:,0])))))
        self.add_axis("delta_phi_wl",
                      "$Cos(\Delta \phi(W, \ell))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.W.delta_phi(events.GoodLeptons[:,0])))))
        self.add_axis("delta_phi_wjet",
                      "$Cos(\Delta \phi(W, Leading-Jet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.W.delta_phi(events.GoodNotBJets[:,0])))))
        self.add_axis("delta_phi_wbjet",
                      "$Cos(\Delta \phi(W, Leading-bJet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.W.delta_phi(events.GoodBJets[:,0])))))
        self.add_axis("delta_phi_topl",
                      "$Cos(\Delta \phi(top, \ell))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.top.delta_phi(events.GoodLeptons[:,0])))))
        self.add_axis("delta_phi_topjet",
                      "$Cos(\Delta \phi(top, Leading-Jet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.top.delta_phi(events.GoodNotBJets[:,0])))))
        self.add_axis("delta_phi_topbjet",
                      "$Cos(\Delta \phi(top, Leading-bJet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.top.delta_phi(events.GoodBJets[:,0])))))
        self.add_axis("delta_phi_tl",
                      "$Cos(\Delta \phi(T, \ell))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.T.delta_phi(events.GoodLeptons[:,0])))))
        self.add_axis("delta_phi_tjet",
                      "$Cos(\Delta \phi(T, Leading-Jet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.T.delta_phi(events.GoodNotBJets[:,0])))))
        self.add_axis("delta_phi_tbjet",
                      "$Cos(\Delta \phi(T, Leading-bJet))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.T.delta_phi(events.GoodBJets[:,0])))))
        self.add_axis("delta_phi_wtop",
                      "$Cos(\Delta \phi(W, top))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.W.delta_phi(events.top[:,0])))))
        self.add_axis("delta_phi_wt",
                      "$Cos(\Delta \phi(W, t))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.W.delta_phi(events.T[:,0])))))
        self.add_axis("delta_phi_ttop",
                      "$Cos(\Delta \phi(t, top))$",
                      bins=10,
                      start=0,
                      stop=1,
                      function = lambda events: ak.flatten(np.cos(abs(events.T.delta_phi(events.top[:,0])))))

        ############################  Multiplicity ###################
        self.add_axis("jet_multiplicity",
                      "Jet Multiplicity",
                      bins=9,
                      start=6,
                      stop=15,
                      function = lambda events: events.nGoodJets)
        self.add_axis("bjet_multiplicity",
                      "BJet Multiplicity",
                      bins=5,
                      start=3,
                      stop=8,
                      function = lambda events: events.nGoodBJets)
                      
        
    def define_histograms(self):
        self.add_histogram("lepton_pt",
                           [self.axes["lepton_pt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("lepton_eta",
                           [self.axes["lepton_eta"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("jet_pt",
                           [self.axes["jet_pt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("jet_eta",
                           [self.axes["jet_eta"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("bjet_pt",
                           [self.axes["bjet_pt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("bjet_eta",
                           [self.axes["bjet_eta"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("met_pt",
                           [self.axes["met_pt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("met_eta",
                           [self.axes["met_eta"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("ht_jets",
                           [self.axes["ht_jets"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("ht_goodJets",
                           [self.axes["ht_goodJets"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("m_tw",
                           [self.axes["m_tw"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("m_w",
                           [self.axes["m_w"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("w_pt",
                           [self.axes["w_pt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("w_eta",
                           [self.axes["w_eta"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("m_top",
                           [self.axes["m_top"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("top_pt",
                           [self.axes["top_pt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("top_eta",
                           [self.axes["top_eta"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("m_t",
                           [self.axes["m_t"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("t_pt",
                           [self.axes["t_pt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("t_eta",
                           [self.axes["t_eta"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )

        #############################     delta          ####################
        self.add_histogram("delta_r_ljet",
                           [self.axes["delta_r_ljet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_lbjet",
                           [self.axes["delta_r_lbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_wl",
                           [self.axes["delta_r_wl"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_wjet",
                           [self.axes["delta_r_wjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_wbjet",
                           [self.axes["delta_r_wbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_topl",
                           [self.axes["delta_r_topl"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_topjet",
                           [self.axes["delta_r_topjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_topbjet",
                           [self.axes["delta_r_topbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_tl",
                           [self.axes["delta_r_tl"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_tjet",
                           [self.axes["delta_r_tjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_tbjet",
                           [self.axes["delta_r_tbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_wtop",
                           [self.axes["delta_r_wtop"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_wt",
                           [self.axes["delta_r_wt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_r_ttop",
                           [self.axes["delta_r_ttop"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )

        ################################        delta phi        #####################
        self.add_histogram("delta_phi_ljet",
                           [self.axes["delta_phi_ljet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_lbjet",
                           [self.axes["delta_phi_lbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_wl",
                           [self.axes["delta_phi_wl"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_wjet",
                           [self.axes["delta_phi_wjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_wbjet",
                           [self.axes["delta_phi_wbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_topl",
                           [self.axes["delta_phi_topl"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_topjet",
                           [self.axes["delta_phi_topjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_topbjet",
                           [self.axes["delta_phi_topbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_tl",
                           [self.axes["delta_phi_tl"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_tjet",
                           [self.axes["delta_phi_tjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_tbjet",
                           [self.axes["delta_phi_tbjet"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_wtop",
                           [self.axes["delta_phi_wtop"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_wt",
                           [self.axes["delta_phi_wt"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("delta_phi_ttop",
                           [self.axes["delta_phi_ttop"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )

        #############################      Multiplicity       #########################
        self.add_histogram("jets_multiplicity",
                           [self.axes["jet_multiplicity"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        self.add_histogram("bjets_multiplicity",
                           [self.axes["bjet_multiplicity"]],
                           ["xsec", "luminosity", "sum_genweight"]
                          )
        
    def add_axis(self,
                 name,
                 label: str,
                 bins: int = None,
                 start: float = None,
                 stop: float = None,
                 type: str = "regular",
                 function: Optional[Callable] = None
                ):
        self.axes[name] = Axis(name, label, bins, start, stop, type, function)
        
    def add_histogram(self,
                      name,
                      axes:List[str],
                      weights= None
                     ):
        self.histograms[name] = Histogram(name, axes, weights)
        
    def get_histogram(self, name):
        return self.histograms[name]
    
    def get_histograms(self):
        return self.histograms
    
