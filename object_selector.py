import awkward as ak
from coffea.nanoevents.methods import vector


class ObjectSelector:
    def __init__(self, events):
        self.events = events     
        
    def selected_electrons(self, channel="1-lep"):

        electrons = self.events.Electron
        if channel == "1-lep":
            pt_mask = electrons.PT >= 25
        elif channel == "2-lep":
            pt_mask = electrons.PT >= 20
        else:
            pt_mask = ak.ones_like(electrons.PT)

        eta_mask = (electrons.Eta < 0.3) & ((electrons.Eta < 1.4442) | (electrons.Eta> 1.566))
 
        iso_mask = electrons.IsolationVarRhoCorr < 0.15      # To be checked
        
        selected_electrons = electrons[pt_mask & iso_mask & eta_mask]
        selected_electrons = ak.with_field(selected_electrons, "e", "flavor")
        selected_electrons = ak.with_field(selected_electrons, selected_electrons.PT, "pt")
        selected_electrons = ak.with_field(selected_electrons, selected_electrons.Eta, "eta")
        selected_electrons = ak.with_field(selected_electrons, selected_electrons.Phi, "phi")
        selected_electrons = ak.with_field(selected_electrons, selected_electrons.mass, "mass")
        
        return selected_electrons
    
    def selected_muons(self, channel="1-lep"):
        
        muons = ak.with_name(self.events.MuonTight, name='PtEtaPhiMLorentzVector')
        if channel == "1-lep":
            pt_mask = muons.PT >= 25
        elif channel == "2-lep":
            pt_mask = muons.PT >= 20
        else:
            pt_mask = ak.ones_like(muons.PT)

        eta_mask = muons.Eta < 2.8 

        iso_mask = muons.IsolationVarRhoCorr < 0.15 # To be checked
        
        selected_muons = muons[pt_mask & iso_mask & eta_mask]
        selected_muons = ak.with_field(selected_muons, "mu", "flavor")
        selected_muons = ak.with_field(selected_muons, selected_muons.PT, "pt")
        selected_muons = ak.with_field(selected_muons, selected_muons.Eta, "eta")
        selected_muons = ak.with_field(selected_muons, selected_muons.Phi, "phi")
        selected_muons = ak.with_field(selected_muons, 0.0*selected_muons.pt, "mass")
        
        return selected_muons
    
    def selected_jets(self, leptons):
        
        selected_jets = self.events.Jet[(self.events.Jet.PT > 25) & (abs(self.events.Jet.Eta) < 5)]
        selected_jets = selected_jets[ak.all(selected_jets.metric_table(leptons) > 0.4, axis=2)]
        # selected_jets = ak.with_field(selected_jets, selected_jets.Eta, "eta")
        # selected_jets = ak.with_field(selected_jets, selected_jets.Phi, "phi")
        
        return selected_jets
    
    def selected_b_jets(self, jets, inverse=False):
        
        eta_mask = jets.Eta <= 2.5 # To be checked
        tag_mask = (jets.BTag == 2) | (jets.BTag == 3) | (jets.BTag == 6) | (jets.BTag == 7)
        btag_mask = eta_mask & tag_mask
        if inverse:
            selected_b_jets = jets[~btag_mask]
        else:
            selected_b_jets = jets[btag_mask]
        
        return selected_b_jets
    
    def select_good_objects(self, channel = "1-lep"):
        
        self.events["GoodElectrons"] = self.selected_electrons(channel)
        self.events["GoodMuons"] = self.selected_muons(channel)
        self.events["GoodLeptons"] = ak.with_name(ak.concatenate((self.events["GoodElectrons"], self.events["GoodMuons"]), axis=1,
                                                  behavior = vector.behavior),
                                                  name='PtEtaPhiMLorentzVector'
                                                 )
        arg = ak.argsort(self.events.GoodLeptons.PT, ascending=False)
        self.events["GoodLeptons"] = self.events["GoodLeptons"][arg]
        self.events["GoodJets"] = self.selected_jets(self.events.GoodLeptons)
        self.events["GoodBJets"] = self.selected_b_jets(self.events.GoodJets)
        self.events["GoodNotBJets"] = self.selected_b_jets(self.events.GoodJets, inverse=True)
        
    def count_good_objects(self):
        
        self.events["nGoodElectrons"] = ak.num(self.events.GoodElectrons)
        self.events["nGoodMuons"] = ak.num(self.events.GoodMuons)
        self.events["nGoodLeptons"] = ak.num(self.events.GoodLeptons)
        self.events["nGoodJets"] = ak.num(self.events.GoodJets)
        self.events["nGoodBJets"] = ak.num(self.events.GoodBJets)
        self.events["nGoodNotBJets"] = ak.num(self.events.GoodNotBJets)