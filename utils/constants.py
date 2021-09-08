
import enum

class AdjacencyMode(str, enum.Enum):
    OneStep = "OneStep",
    Partial = "Partial"

class ModelType(str, enum.Enum):
      GATV1 = "GAT-V1",
      GATV2 = "GAT-V2", 
      TRANS = "TRANS"
      
class Dataset(str, enum.Enum):
    OGBN_ARXIV = 'ogbn-arxiv',
    OGBN_PRODUCTS = 'ogbn-products',
    OGBN_PROTEINS = 'ogbn-proteins',
    OGBN_MAG = 'ogbn-mag'
    
datasetMapping = {
    Dataset.OGBN_ARXIV: "ogbn-arxiv",
    Dataset.OGBN_PRODUCTS: "ogbn-products",
    Dataset.OGBN_PROTEINS: "ogbn-proteins",
    Dataset.OGBN_MAG: "ogbn-mag"
}
    