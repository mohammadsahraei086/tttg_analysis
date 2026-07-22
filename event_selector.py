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
    
    def select_n_lep_events(self, channel="1-lep"):
        selection = PackedSelection()
        selection.add(channel, self.events.nGoodLeptons == int(channel[0]))
        nlep_events = self.events[selection.all(channel)]
        
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
    
        
        # pt = MET_pt
        # phi = MET_phi
        # theta = np.arctan2(pt, pz_nu)
        # eta = -np.log(np.tan(theta / 2))
        # m = np.sqrt(np.maximum(E_nu**2 - (MET_px**2 + MET_py**2 + pz_nu**2), 0))
        # nu_p4 = ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": m},with_name="PtEtaPhiMLorentzVector")
    
        return nu_p4

    def define_variables_before_selection(self, events, channel="1-lep"):
        
        events["HT_Jets"] = ak.sum(events.Jet.pt, axis=1)
        events["HT_GoodJets"] = ak.sum(events.GoodJets.pt, axis=1)

    def define_variables_after_selection(self, events, channel="1-lep"):
        
        events["W_T"] = np.sqrt(2*events.GoodLeptons.PT*events.MissingET.MET*(1-np.cos(events.GoodLeptons.delta_phi(events.MissingET))))
        events["neutrino"] = self.calculateNu4vec(events.GoodLeptons, events.MissingET)
        events["W"] = events.GoodLeptons.add(events.neutrino)
        events["top"] = events.W.add(events.GoodBJets[:,0])
        events["T"] = events.top.add(events.GoodNotBJets[:, 0])
        
        
    def select_good_events(self, channel="1-lep"):
        selection = PackedSelection()
        cutflow = {}
        cutflow["primary"] = len(self.events)
        selected_events = self.select_n_lep_events(channel)
        cutflow[channel] = len(selected_events)
        self.define_variables_before_selection(selected_events, channel)
        
        if channel == "1-lep":
            
            selection.add("nJet", selected_events.nGoodJets >= 6)
            selection.add("nBJet", selected_events.nGoodBJets >= 3)
            selection.add("nNotBJet", selected_events.nGoodNotBJets >= 1)
            selection.add("HT_jets",  selected_events.HT_Jets > 600) # 
            selection.add("MET", selected_events.MissingET.MET >= 30)

            mask = ak.Array([True] * len(selected_events))
            for name in selection.names:
                new_mask = selection.all(name)
                cutflow[name] = ak.sum(mask & new_mask)
                mask = mask & new_mask 

            selected_events = selected_events[selection.all("nJet", "nBJet", "nNotBJet", "HT_jets", "MET")]

        elif channel == "2-lep":
            selection.add("nJet", selected_events.nGoodJets >= 5)
            selection.add("nBJet", selected_events.nGoodBJets >= 3)
            selection.add("HT", ak.sum(selected_events.GoodJets.pt, axis=1) > 500)
            selection.add("MET", selected_events.MissingET.MET >= 30)

            mask_flavor = selected_events.GoodLeptons[:, 0].flavor == selected_events.GoodLeptons[:, 1]
            mask_charge = (selected_events.GoodLeptons[:, 0].charge + selected_events.GoodLeptons[:, 1].charge) == 0
            mask_invMass = ((selected_events.GoodLeptons[:, 0] + selected_events.GoodLeptons[:, 1]).mass - 91.1876) <= 15
            selection.add("lepInvMass", ~(mask_flavor & mask_charge & mask_invMass)) 

            mask = ak.Array([True] * len(selected_events))
            for name in selection.names:
                new_mask = selection.all(name)
                cutflow[name] = ak.sum(mask & new_mask)
                mask = mask & new_mask 

            selected_events = selected_events[selection.all("nJet","nBJet", "HT", "MET", "lepInvMass")]

        else:
            selection.add("nJet", selected_events.nGoodJets >= 4)
            selection.add("nBJet", selected_events.nGoodBJets >= 3)
            selection.add("MET", selected_events.MissingET.MET >= 40)

            mask = ak.Array([True] * len(selected_events))
            for name in selection.names:
                new_mask = selection.all(name)
                cutflow[name] = ak.sum(mask & new_mask)
                mask = mask & new_mask 

            selected_events = selected_events[selection.all("nJet","nBJet", "MET")]

        self.define_variables_after_selection(selected_events, channel)
                            
        return selected_events, cutflow


        # selection.add("leadingLepPT", selected_events.GoodLeptons[:, 0].pt > 25)
        #     selection.add("OCLep", (selected_events.GoodLeptons[:, 0].charge + selected_events.GoodLeptons[:, 1].charge) == 0)
        #     selection.add("lepInvariantMass", (selected_events.GoodLeptons[:, 0] + selected_events.GoodLeptons[:, 1]).mass > 20)
        #     selection.add("onePhoton", selected_events.nGoodPhotons == 1)
        #     selection.add("atLeastOneBJet", selected_events.nGoodBJets >= 1)
    
        #     # Add selection for different channels
        #     selection.add("emu", selected_events.GoodLeptons.flavor[:, 0] != selected_events.GoodLeptons.flavor[:, 1])
        #     selection.add("ee", (selected_events.GoodLeptons.flavor[:, 0] == "e") & (selected_events.GoodLeptons.flavor[:, 1]=="e"))
        #     selection.add("mumu", (selected_events.GoodLeptons.flavor[:, 0] == "mu") & (selected_events.GoodLeptons.flavor[:, 1]=="mu"))