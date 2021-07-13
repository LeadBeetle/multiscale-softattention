
import enum

class AdjacencyMode(enum.Enum):
    OneStep = 0,
    Partial = 1

class ModelType(enum.Enum):
      GATV1 = 0,
      GATV2 = 1, 
      TRANS = 2
      
class Dataset(enum.Enum):
    OGBN_ARXIV = 0,
    OGBN_PRODUCTS = 1, 
    OGBN_PROTEINS = 2,
    OGBN_MAG = 3
    
datasetMapping = {
    Dataset.OGBN_ARXIV: "ogbn-arxiv",
    Dataset.OGBN_PRODUCTS: "ogbn-products",
    Dataset.OGBN_PROTEINS: "ogbn-proteins",
    Dataset.OGBN_MAG: "ogbn-mag"
}
    