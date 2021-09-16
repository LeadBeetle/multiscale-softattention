from utils.experimentor_base import *
from utils.experimentor_proteins import * 
from utils.experimentor_planetoid import * 
from utils.constants import *

def getExperimentor(dataset_name): 
    isProteins = dataset_name == Dataset.OGBN_PROTEINS
    isPlanetoid = dataset_name in [Dataset.CORA, Dataset.PUBMED, Dataset.CITESEER]

    exp = None
    if isProteins:
        exp = Experimentor_Proteins
    elif isPlanetoid:
        exp = Experimentor_Planetoid
    else: 
        exp = Experimentor
    assert(exp is not None)
    return exp


def runExperiments(config, models, datasets, degrees):
    for model in [ModelType.GATV1, ModelType.GATV2, ModelType.TRANS]:
        config["model_type"] = model
        for dataset in [Dataset.CORA, Dataset.CITESEER, Dataset.PUBMED]:
            config["dataset_name"] = dataset
            if dataset == Dataset.PUBMED:
                config["lr"] = 0.01
                config["num_heads"] = 8
            else: 
                config["lr"] = 0.005
                config["num_heads"] = 1
                
            for degree in [1, 2, 3, 4]:
                config["nbor_degree"] = degree
                for isSparse in [False, True]:
                    config["sparse"] = isSparse
                    experimentor = getExperimentor(config["dataset_name"])(config)
                    experimentor.run_wrapper()