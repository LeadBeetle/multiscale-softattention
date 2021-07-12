
import enum

class AdjacencyMode(enum.Enum):
    OneStep = 0,
    Partial = 1
    
class Dataset(enum.Enum):
    OGBN_ARXIV = "ogbn-arxiv",
    OGBN_PRODUCTS = "ogbn-products", 
    OGBN_PROTEINS = "ogbn-proteins",
    OGBN_MAG = "ogbn-mag"
    
datasetMapping = {
    Dataset.OGBN_ARXIV: "ogbn-arxiv",
    Dataset.OGBN_PRODUCTS: "ogbn-products",
    Dataset.OGBN_PROTEINS: "ogbn-proteins",
    Dataset.OGBN_MAG: "ogbn-mag"
}
    