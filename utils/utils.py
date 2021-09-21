import torch
torch.manual_seed(43)
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor, set_diag
from torch_geometric.utils import remove_self_loops, add_self_loops
from utils.constants import * 
import logging
                
def one_step(edge_index, x, num_nodes, device):
    edge_weight = None
    if x>1: 
        edge_weight = torch.ones(edge_index.size(1))
        adj = to_dense_adj(edge_index)[0]
        for k in range(2,x+1):
            adj_k = (torch.linalg.matrix_power(adj, k)>0)*1 - adj
            
            edge_index_k = (adj_k > 0).nonzero().t()
            edge_weight_k = 1/k * torch.ones(edge_index_k.size(1))
            edge_index = torch.cat((edge_index, edge_index_k), dim=1)
            edge_weight = torch.cat((edge_weight, edge_weight_k), dim=0)
    if edge_weight != None:
        edge_weight = edge_weight.to(device)
    return edge_index, edge_weight


def one_step_sparse(edge_index: SparseTensor, x, num_nodes, device):
    size = torch.Size([num_nodes, num_nodes])
    edge_index = edge_index.sparse_resize((num_nodes, num_nodes))
    adj = SparseTensor.to_torch_sparse_coo_tensor(edge_index)
    if x>1: 
        adj_powers = [adj]
        for k in range(2, x+1):
            adj_powers.append(torch.sparse.mm(adj_powers[k-2], adj_powers[0]))
            adj_k = adj_powers[k-1] - adj_powers[0]
            adj_k = torch.sparse_coo_tensor(adj_k._indices(), 1/k * (adj_k._values()>0)*1, size = size)
            adj = adj + adj_k 
    return SparseTensor.from_torch_sparse_coo_tensor(adj).t(), None


def getResultFileName(config): 
    model  = config["model_type"]
    degree = config["nbor_degree"]
    sparse = config["sparse"]

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

    sparsePart = "s" if sparse == True else "ns"

    return "_".join([modelPart, degreePart, sparsePart])