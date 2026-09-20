from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import vector
import awkward as ak
import numpy as np

class EventSelector:
    def __init__(self, events):
        self.events = events

    def add_trigger_selection(self):
        pass

    def primary_skim(self):
        self.add_trigger_selection()

        return self._selection("trigger")
    
    def select_n_lep_events(self, cat="noEFT"):
        selection = PackedSelection()
        selection.add("nlep=1", (self.events.nGoodLeptons == 1) & (self.events.nLooseLeptons == 1))
        nlep_events = self.events[selection.all("nlep=1")]
        
        
        return nlep_events

    def calculateNu4vec(self, lepton, MET):
        # MET components
        MET_pt = MET.MET 
        MET_phi = MET.Phi
        MET_px = MET_pt * np.cos(MET_phi)
        MET_py = MET_pt * np.sin(MET_phi) 
        # Lepton components
        lep_m = lepton.mass
        lep_eta = lepton.eta
        lep_pt = lepton.pt
        lep_phi = lepton.phi
        lep_py = lep_pt * np.sin(lep_phi)
        lep_px = lep_pt * np.cos(lep_phi)
        lep_pz = lep_pt * np.sinh(lep_eta)
        lep_E = np.sqrt(lep_px**2 + lep_py**2 + lep_pz**2 + lep_m**2)
        # Constants
        MW = 80.38  # W boson mass in GeV
    
    
        #Discriminant
        A = pow(lep_pz,2)-pow(lep_E,2)
        alpha = pow(MW,2)-pow(lep_m,2) + 2 * (lep_px * MET_px + lep_py * MET_py)
        B = alpha * lep_pz
        C = (- pow(lep_E,2) * pow(MET_pt,2) ) + np.divide(pow(alpha,2),4)
        dis = (pow(B,2) - (4*A*C))  # b2-4AC
    
        condition = dis >= 0
        
        root = np.sqrt(ak.where(condition, dis, ak.zeros_like(dis)))
        root1 = np.divide(- B - root, 2*A)
        root2 = np.divide(- B + root, 2*A)
        pz_nu = ak.where(np.abs(root1) < np.abs(root2), root1, root2)
        E_nu = np.sqrt(MET_pt**2 + pz_nu**2)  
    
        real_root = ak.where(condition, ak.zeros_like(dis), -B/(2*A)) 
        pz_nu = ak.where(condition, pz_nu, real_root)
        E_nu = np.sqrt(MET_pt**2 + pz_nu**2)
    
        nu_p4 = ak.zip(
            {
                "x": MET_px,
                "y": MET_py,
                "z": pz_nu,
                "t": E_nu,  # t is the energy/time component
            },
            with_name="LorentzVector",  # This gives you the Cartesian Lorentz vector
            behavior=vector.behavior
        )
    
        return nu_p4

    def define_variables_before_selection(self, events, cat="noEFT"):
        
        events["HT_Jets"] = ak.sum(events.Jet.pt, axis=1)
        events["HT_GoodJets"] = ak.sum(events.GoodJets.pt, axis=1)
        events["S_T"] = ak.sum(events.GoodJets.pt, axis=1) + ak.sum(events.GoodLeptons.pt, axis=1) + events.MissingET.MET
        if ak.any(events.Event.X1*events.Event.X2, axis=0) == 0:
            raise ValueError("Center of Mass is energy is zero")
        else:
            events["s_hat"] = np.sqrt(events.Event.X1*events.Event.X2)*14000

    def define_variables_after_selection(self, events, cat="noEFT"):
        
        events["W_T"] = np.sqrt(2*events.GoodLeptons.PT*events.MissingET.MET*(1-np.cos(events.GoodLeptons.delta_phi(events.MissingET))))
        events["neutrino"] = self.calculateNu4vec(events.GoodLeptons, events.MissingET)
        events["W"] = events.GoodLeptons.add(events.neutrino)
        events["top"] = events.W.add(events.GoodBJets[:,0])
        events["T"] = events.top.add(events.GoodNotBJets[:, 0])
        
    def select_good_events(self, cat="noEFT"):
        selection = PackedSelection()
        cutflow = {}
        cutflow["nevents"] = {}
        cutflow["yield"] = {}
        weight = (self.events.metadata["xsec"] * 3000)/self.events.metadata["nevents"]
        cutflow["yield"]["primary"] = len(self.events) * weight
        cutflow["nevents"]["primary"] = len(self.events)
        
        selected_events = self.select_n_lep_events(cat)
        cutflow["nevents"]["nlep=1"] = len(selected_events)
        cutflow["yield"]["nlep=1"] = len(selected_events) * weight
        self.define_variables_before_selection(selected_events, cat)

        selection.add("MET", selected_events.MissingET.MET >= 30)
        selection.add("nJet", selected_events.nGoodJets >= 8)
        selection.add("nBJet", selected_events.nGoodBJets >= 2)
        selection.add("nNotBJet", selected_events.nGoodNotBJets >= 1)
        # selection.add("HT_jets",  selected_events.HT_Jets > 600) # 

        mask = ak.Array([True] * len(selected_events))
        for name in selection.names:
            if name=="nNotBJet":
                continue
            new_mask = selection.all(name)
            cutflow["yield"][name] = ak.sum(mask & new_mask)*weight
            cutflow["nevents"][name] = ak.sum(mask & new_mask)
            mask = mask & new_mask 
        
        if "Signal_" in self.events.metadata["dataset"]:
            if cat == "noEFT":
                selected_events = selected_events[selection.all("MET", "nJet", "nBJet", "nNotBJet")]
            elif cat == "sEFT":
                selection.add("s<L", selected_events.S_T < 5000)
                selected_events = selected_events[selection.all("MET", "nJet", "nBJet", "nNotBJet", "s<L")]
                cutflow["yield"]["s<l"] = len(selected_events)*weight
                cutflow["nevents"]["s<l"] = len(selected_events)
            else:
                selection.add("s_hat<L", selected_events.s_hat < 5000)
                selected_events = selected_events[selection.all("MET", "nJet", "nBJet", "nNotBJet", "s_hat<L")]
                cutflow["yield"]["s_hat<l"] = len(selected_events)*weight
                cutflow["nevents"]["s_hat<l"] = len(selected_events)
        else:
            selected_events = selected_events[selection.all("MET", "nJet", "nBJet", "nNotBJet")]
                

        self.define_variables_after_selection(selected_events, cat)
                            
        return selected_events, cutflow