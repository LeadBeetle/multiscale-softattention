import torch
torch.manual_seed(43)
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor
from utils.constants import *    

def one_step(edge_index, x, num_nodes, device):
    edge_weight = None
    if x>1: 
        edge_weight = torch.ones(edge_index.size(1))
        adj = to_dense_adj(edge_index)[0]
        current_adj_power = adj.detach().clone()
        for k in range(2,x+1):
            #print("k=", k)
            current_adj_power = (torch.mm(current_adj_power, adj)>0)*1 - adj
            edge_index_k = (current_adj_power > 0).nonzero().t()
            edge_weight_k = 1/k * torch.ones(edge_index_k.size(1))
            edge_index = torch.cat((edge_index, edge_index_k), dim=1)
            edge_weight = torch.cat((edge_weight, edge_weight_k), dim=0)
    if edge_weight != None and edge_index.device() == device:
        edge_weight = edge_weight.to(device)
    return edge_index, edge_weight


def one_step_sparse(edge_index: SparseTensor, x, num_nodes, device):
    if x>1: 
        size = torch.Size([num_nodes, num_nodes])
        edge_index = edge_index.sparse_resize((num_nodes, num_nodes))
        base_adj = SparseTensor.to_torch_sparse_coo_tensor(edge_index)
        current_adj_power = base_adj.detach().clone()
        adj = base_adj.detach().clone()
        for k in range(2, x+1):
            #print("k=", k)
            current_adj_power = torch.sparse.mm(current_adj_power, base_adj)
            adj_k = current_adj_power - base_adj
            adj_k = torch.sparse_coo_tensor(adj_k._indices(), 1/k * (adj_k._values()>0)*1, size = size)
            adj = adj + adj_k 
        return SparseTensor.from_torch_sparse_coo_tensor(adj), None
    else: 
        return edge_index, None


def getResultFileName(config): 
    model  = config["model_type"]
    degree = config["nbor_degree"]
    sparse = config["sparse"]
    num_layers = config["num_of_layers"]
    modelPart = ""

    if model == ModelType.GATV1:
        modelPart = "GatV1"
    elif model == ModelType.GATV2:
        modelPart = "GatV2"
    elif model == ModelType.TRANS:
        modelPart = "Trans"
    assert(modelPart != "")
    
    degreePart = "".join(["K", str(degree)])
    assert(degreePart != "")

    assert(num_layers > 0)
    layerPart = "D" + str(num_layers)
    sparsePart = "s" if sparse == True else "ns"
    
    return "_".join([modelPart, degreePart, layerPart, sparsePart])