import time
import copy
import awkward as ak
import numpy as np

from coffea import processor
from coffea.util import save
from coffea.nanoevents import DelphesSchema
from coffea.processor import column_accumulator

from hist_manager import HistManager
from object_selector import ObjectSelector
from event_selector import EventSelector
from fileset import *

fileset = fileset

class Analysis(processor.ProcessorABC):
    
    def __init__(self):
        self.hist_manager = HistManager()
        self.hist_manager.define_axes()
        self.hist_manager.define_histograms()
        self.histograms = self.hist_manager.get_histograms()
        self.categories = ["1-lep"]  # "2-lep", "3-lep"
    
    def define_output_layout(self):
        output = {}
        output["metadata"] = {}
        output["nEvents"] = {}
        output["nEvents"]["primary"] = {}
        output["nEvents"]["selected"] = {}
        output["nEvents"]["cutflow"] = {}
        output["hists"] = {}
        for cat in self.categories:
            output["nEvents"]["selected"][cat] = {}
            output["nEvents"]["cutflow"][cat] = {}
            output["hists"][cat] = {}
            for hist in self.histograms:
                output["hists"][cat][hist] = {}
                    
        return output

    def store_features_for_NN(self, events, dts):
        self.output["features"] = {}
        self.output["features"][dts] = {}

        self.output["features"][dts]["lepton_pt"] = column_accumulator(ak.to_numpy(events.GoodLeptons.PT))
        self.output["features"][dts]["lepton_eta"] = column_accumulator(ak.to_numpy(events.GoodLeptons.Eta))
        self.output["features"][dts]["jet_shape"] = column_accumulator(ak.to_numpy(ak.num(events.GoodJets.PT)))
        self.output["features"][dts]["bjet_shape"] = column_accumulator(ak.to_numpy(ak.num(events.GoodBJets.PT)))
        self.output["features"][dts]["jet_pt"] = column_accumulator(ak.to_numpy(ak.fill_none(ak.pad_none(events.GoodJets.PT, 6, axis=1, clip=True),
                                                                                             0.0)))
        self.output["features"][dts]["jet_eta"] = column_accumulator(ak.to_numpy(ak.fill_none(ak.pad_none(events.GoodJets.eta, 6, axis=1, clip=True),
                                                                                             0.0)))
        self.output["features"][dts]["bjet_pt"] = column_accumulator(ak.to_numpy(ak.fill_none(ak.pad_none(events.GoodBJets.PT, 3, axis=1, clip=True),
                                                                                             0.0)))
        self.output["features"][dts]["bjet_eta"] = column_accumulator(ak.to_numpy(ak.fill_none(ak.pad_none(events.GoodBJets.eta, 3, axis=1, clip=True),
                                                                                             0.0)))
        self.output["features"][dts]["met_pt"] = column_accumulator(ak.to_numpy(events.MissingET.MET))
        self.output["features"][dts]["met_eta"] = column_accumulator(ak.to_numpy(events.MissingET.eta))
        self.output["features"][dts]["ht_jets"] = column_accumulator(ak.to_numpy(events.HT_Jets))
        self.output["features"][dts]["ht_goodJets"] = column_accumulator(ak.to_numpy(events.HT_GoodJets))
        self.output["features"][dts]["m_wt"] = column_accumulator(ak.to_numpy(events.W_T))
        self.output["features"][dts]["W_pt"] = column_accumulator(ak.to_numpy(events.W.pt))
        self.output["features"][dts]["W_eta"] = column_accumulator(ak.to_numpy(events.W.eta))
        self.output["features"][dts]["top_mass"] = column_accumulator(ak.to_numpy(events.top.mass))
        self.output["features"][dts]["top_pt"] = column_accumulator(ak.to_numpy(events.top.pt))
        self.output["features"][dts]["top_eta"] = column_accumulator(ak.to_numpy(events.top.eta))
        self.output["features"][dts]["t_mass"] = column_accumulator(ak.to_numpy(events.T.mass))
        self.output["features"][dts]["t_pt"] = column_accumulator(ak.to_numpy(events.T.pt))
        self.output["features"][dts]["t_eta"] = column_accumulator(ak.to_numpy(events.T.eta))

        ##############################     delta r      #############################

        self.output["features"][dts]["delta_r_ljet"] = column_accumulator(ak.to_numpy(events.GoodLeptons.delta_r(events.GoodNotBJets[:,0])))
        self.output["features"][dts]["delta_r_lbjet"] = column_accumulator(ak.to_numpy(events.GoodLeptons.delta_r(events.GoodBJets[:,0])))
        self.output["features"][dts]["delta_r_wl"] = column_accumulator(ak.to_numpy(events.W.delta_r(events.GoodLeptons[:,0])))
        self.output["features"][dts]["delta_r_wjet"] = column_accumulator(ak.to_numpy(events.W.delta_r(events.GoodNotBJets[:,0])))
        self.output["features"][dts]["delta_r_wbjet"] = column_accumulator(ak.to_numpy(events.W.delta_r(events.GoodBJets[:,0])))
        self.output["features"][dts]["delta_r_topl"] = column_accumulator(ak.to_numpy(events.top.delta_r(events.GoodLeptons[:,0])))
        self.output["features"][dts]["delta_r_topjet"] = column_accumulator(ak.to_numpy(events.top.delta_r(events.GoodNotBJets[:,0])))
        self.output["features"][dts]["delta_r_topbjet"] = column_accumulator(ak.to_numpy(events.top.delta_r(events.GoodBJets[:,0])))
        self.output["features"][dts]["delta_r_tl"] = column_accumulator(ak.to_numpy(events.T.delta_r(events.GoodLeptons[:,0])))
        self.output["features"][dts]["delta_r_tjet"] = column_accumulator(ak.to_numpy(events.T.delta_r(events.GoodNotBJets[:,0])))
        self.output["features"][dts]["delta_r_tbjet"] = column_accumulator(ak.to_numpy(events.T.delta_r(events.GoodBJets[:,0])))
        self.output["features"][dts]["delta_r_wtop"] = column_accumulator(ak.to_numpy(events.W.delta_r(events.top[:,0])))
        self.output["features"][dts]["delta_r_wt"] = column_accumulator(ak.to_numpy(events.W.delta_r(events.T[:,0])))
        self.output["features"][dts]["delta_r_ttop"] = column_accumulator(ak.to_numpy(events.T.delta_r(events.top[:,0])))

        ###############################     delta phi     #############################

        self.output["features"][dts]["delta_phi_ljet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.GoodLeptons.delta_phi(events.GoodNotBJets[:,0])))))
        self.output["features"][dts]["delta_phi_lbjet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.GoodLeptons.delta_phi(events.GoodBJets[:,0])))))
        self.output["features"][dts]["delta_phi_wl"] = column_accumulator(ak.to_numpy(np.cos(abs(events.W.delta_phi(events.GoodLeptons[:,0])))))
        self.output["features"][dts]["delta_phi_wjet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.W.delta_phi(events.GoodNotBJets[:,0])))))
        self.output["features"][dts]["delta_phi_wbjet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.W.delta_phi(events.GoodBJets[:,0])))))
        self.output["features"][dts]["delta_phi_topl"] = column_accumulator(ak.to_numpy(np.cos(abs(events.top.delta_phi(events.GoodLeptons[:,0])))))
        self.output["features"][dts]["delta_phi_topjet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.top.delta_phi(events.GoodNotBJets[:,0])))))
        self.output["features"][dts]["delta_phi_topbjet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.top.delta_phi(events.GoodBJets[:,0])))))
        self.output["features"][dts]["delta_phi_tl"] = column_accumulator(ak.to_numpy(np.cos(abs(events.T.delta_phi(events.GoodLeptons[:,0])))))
        self.output["features"][dts]["delta_phi_tjet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.T.delta_phi(events.GoodNotBJets[:,0])))))
        self.output["features"][dts]["delta_phi_tbjet"] = column_accumulator(ak.to_numpy(np.cos(abs(events.T.delta_phi(events.GoodBJets[:,0])))))
        self.output["features"][dts]["delta_phi_wtop"] = column_accumulator(ak.to_numpy(np.cos(abs(events.W.delta_phi(events.top[:,0])))))
        self.output["features"][dts]["delta_phi_wt"] = column_accumulator(ak.to_numpy(np.cos(abs(events.W.delta_phi(events.T[:,0])))))
        self.output["features"][dts]["delta_phi_ttop"] = column_accumulator(ak.to_numpy(np.cos(abs(events.T.delta_phi(events.top[:,0])))))

        #####################################      Multiplicity          ###########################
        
        self.output["features"][dts]["njets"] = column_accumulator(ak.to_numpy(events.nGoodJets))
        self.output["features"][dts]["nbjets"] = column_accumulator(ak.to_numpy(events.nGoodBJets))
        

    def process(self, events):
        dataset = events.metadata["dataset"]
        nevents = events.metadata["nevents"]
        self.output = self.define_output_layout()
        self.events = events
        n_primary = len(self.events)
        self.output["nEvents"]["primary"][dataset] = n_primary

        object_selector = ObjectSelector(self.events)
        event_selector = EventSelector(self.events)

        for cat in self.categories:
            object_selector.select_good_objects(cat)
            object_selector.count_good_objects()
            selected_events, cutflow = event_selector.select_good_events(cat)
            self.output["nEvents"]["cutflow"][cat][dataset] = cutflow
            if len(selected_events) == 0:
                continue
            self.store_features_for_NN(selected_events, dataset)
            self.output["nEvents"]["selected"][cat][dataset] = len(selected_events)
            for name, hist in self.histograms.items():
                # hist_copy = copy.deepcopy(hist)
                hist.fill(selected_events, nevents)
                self.output["hists"][cat][name][dataset] = copy.deepcopy(hist.get_histogram())
                hist.reset_histogram()
        
        return self.output

    def postprocess(self, accumulator):
        for dataset, value in fileset.items():
            accumulator["metadata"][dataset] = value["metadata"]


#####################################################################################################################
def main():
    # client = Client()

    tstart = time.time()
    
    futures_run = processor.Runner(
        executor = processor.FuturesExecutor(compression=None, workers=20),
        schema=DelphesSchema,
        # maxchunks=10,
    )

    out = futures_run(
        fileset,
        treename="Delphes",
        processor_instance=Analysis(),
    )
    
    # iterative_run = processor.Runner(
    #     executor = processor.IterativeExecutor(compression=None),
    #     schema=DelphesSchema,
    #     chunksize=10000,
    #     # maxchunks=None,
    # )
    
    # out = iterative_run(
    #     fileset,
    #     treename="Delphes",
    #     processor_instance=Analysis(),
    # )
    # print(out)
    save(out, 'output/output.coffea')
    
    elapsed = time.time() - tstart
    # print(elapsed)

if __name__ == "__main__":
    main()