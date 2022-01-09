import matplotlib.pyplot as plt
import numpy as np
from utils.constants import *
import os 
import os.path as osp
import json 
from mergedeep import merge

resultPath = osp.abspath("results")
def visualize(): 
    pass


def ft(string):
    string = str(string).lower()
    string = ''.join(e for e in string if e.isalnum())
    string = string.replace("_", "")
    string = string.replace("-", "")
    return string    

class Compressor(): 
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_accs      = []
        self.test_accs       = []
        self.val_accs        = []
        self.train_stds      = []
        self.test_stds       = []
        self.val_stds        = []
        self.train_time_avgs = []
        self.epochs          = []
        self.num_of_layers   = 0

    def compressAll(self):
        for dataset in [Dataset.CORA, Dataset.PUBMED, Dataset.CITESEER, Dataset.OGBN_PRODUCTS]:
            for model in [ModelType.GATV1, ModelType.GATV2, ModelType.TRANS]:
                for aggr_mode in [AggrMode.NONE, AggrMode.MEAN, AggrMode.MAX]:
                    for num_of_layers in [2,3,4]:
                        self.compressResults(dataset, model, num_of_layers, aggr_mode) 
        

    def setCompressed(self, dataset, model, num_of_layers, aggr_mode): 
        if len(self.test_accs) == 0: return None

        res = {
            dataset.name: 
            { 
                model.name: 
                {   
                    aggr_mode.name: {
                        "NumOfLayers_" + str(num_of_layers): {
                            "train_accs ": self.train_accs,
                            "test_accs"  : self.test_accs,
                            "val_accs"   : self.val_accs,
                            "train_stds" : self.train_stds,
                            "test_stds"  : self.test_stds,
                            "val_stds"   : self.val_stds,
                            "train_times": self.train_time_avgs,
                            "epochs"     : self.epochs
                        } 
                    }
                    
                }
            }
        }
        return res

    def compressResults(self, dataset: Dataset, model: ModelType, num_of_layers, aggr_mode):
        self.folder = osp.join(resultPath, dataset.name)
        self.reset()
        for path, _, files in os.walk(self.folder):
            for file in files:
                if not "checkpoint" in file:
                    self.getDataFromResultFile(path, file, dataset, model, num_of_layers, aggr_mode)
        filename = osp.join(self.folder, "CompressedResults" + ".json")
        res = self.setCompressed(dataset, model, num_of_layers, aggr_mode)
        if res is not None:
            currentState = None
            if osp.exists(filename):
                f = open(filename)
                currentState = merge(json.load(f), res)
                f.close()
            else:
                currentState = res

            with open(filename, 'w') as f:
                json.dump(currentState, f, ensure_ascii=False, indent=4)

    def getDataFromResultFile(self, path, file, dataset, model, num_of_layers, aggr_mode):
        if not "Compressed" in file:
            f = open(osp.join(path, file))
            
            results = json.load(f)
            matchDataset = ft(results["dataset_name"]) == ft(dataset.name)
            matchModel = ft(results["model_type"]) == ft(model.name)
            matchNumLayers = int(results["num_of_layers"]) == int(num_of_layers)

            _aggrMode = AggrMode.NONE.name
            if "aggr_mode" in results:
                _aggrMode = ft(results["aggr_mode"])
            matchAggr = ft(_aggrMode) == ft(aggr_mode.name)
            
            if matchDataset and matchModel and matchNumLayers and matchAggr:
                # if ft(results["dataset_name"]) == "pubmed" and ft(dataset.name)=="pubmed" and int(num_of_layers)==2 and int(results["num_of_layers"])==2 and ft(results["model_type"])=="gatv1" and ft(model.name) == "gatv1" and ft(_aggrMode) == "none"  and ft(_aggrMode) == "none":
                #     print(file, path, results)

                self.train_accs.append(results["train_acc_mean"])
                self.val_accs.append(results["val_acc_mean"])
                self.test_accs.append(results["test_acc_mean"])
                self.train_stds.append(results["train_acc_std"])
                self.val_stds.append(results["val_acc_std"])
                self.test_stds.append(results["test_acc_std"])
                if "train_time_avg" in results.keys():
                    train_time_avg = results["train_time_avg"]
                    self.train_time_avgs.append(train_time_avg)
                if "num_epochs_avg" in results.keys():
                    epoch = results["num_epochs_avg"]
                    if epoch is not None:
                        self.epochs.append(epoch)
            f.close()
            