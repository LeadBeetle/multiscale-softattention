from utils.experimentor_base import *
from utils.experimentor_proteins import * 
from utils.experimentor_planetoid import * 
from utils.constants import *
from utils.configs import * 
from mergedeep import merge

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

def setConfig(dataset, model, config):
    specificConfig = None
    if dataset == Dataset.PUBMED:
        specificConfig = merge(base_config, pub_config)
    elif dataset in [Dataset.CORA, Dataset.CITESEER]: 
        specificConfig = merge(base_config, coracite_config)
    elif dataset == Dataset.OGBN_ARXIV:
        specificConfig = merge(base_config, arxiv_config)
        if model == ModelType.TRANS:
            specificConfig = merge(specificConfig, arxprod_trans_config)
    elif dataset == Dataset.OGBN_PRODUCTS:
        specificConfig = merge(base_config, products_config)
        if model == ModelType.TRANS:
            specificConfig = merge(specificConfig, arxprod_trans_config)
    elif dataset == Dataset.OGBN_PROTEINS:
        specificConfig = merge(base_config, proteins_config)
        if model == ModelType.TRANS:
            specificConfig = merge(specificConfig, proteins_trans_config)
    return merge(config, specificConfig)

def runExperiments(models, datasets, degrees):
    config = {}
    for model in models:
        config["model_type"] = model
        for dataset in datasets:
            config["dataset_name"] = dataset
            config = setConfig(dataset, model, config)
                
            for degree in degrees:
                config["nbor_degree"] = degree
                for isSparse in [False, True]:
                    config["sparse"] = isSparse
                    experimentor = getExperimentor(config["dataset_name"])(config)
                    experimentor.run_wrapper()