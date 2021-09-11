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