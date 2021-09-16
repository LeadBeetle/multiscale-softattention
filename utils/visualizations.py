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

    def compressAll(self):
        for dataset in [Dataset.CORA, Dataset.PUBMED, Dataset.CITESEER]:
            for model in [ModelType.GATV1, ModelType.GATV2, ModelType.TRANS]:
                for sparse in [False, True]:

                    self.compressResults(dataset, model, sparse) 
        

    def setCompressed(self, dataset, model, sparseStr): 
        res = {
            dataset.name: 
            { 
                model.name: 
                {
                    sparseStr: {
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
        return res

    def compressResults(self, dataset: Dataset, model: ModelType, sparse: bool):
        self.folder = osp.join(resultPath, dataset.name)#
        self.reset()

        for _, _, files in os.walk(self.folder):
            for file in files:
                self.getDataFromResultFile(file, dataset, sparse, model)
        filename = osp.join(self.folder, "CompressedResults" + ".json")
        sparseStr = "sparse" if sparse else "dense"

        res = self.setCompressed(dataset, model, sparseStr)
        
        currentState = None
        if osp.exists(filename):
            f = open(filename)
            currentState = merge(json.load(f), res)
            f.close()
        else:
            currentState = res

        with open(filename, 'w') as f:
            json.dump(currentState, f, ensure_ascii=False, indent=4)

    def getDataFromResultFile(self, file, dataset, sparse, model):
        if not "Compressed" in file:
            f = open(osp.join(self.folder, file))
            
            results = json.load(f)
            
            if ft(results["dataset_name"]) == ft(dataset.name) and ft(results["sparse"]) == ft(sparse) and ft(results["model_type"]) == ft(model.name):
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

