import torch
from torch_geometric.utils import to_dense_adj


def adj_times_x(adj, x):
    return torch.linalg.matrix_power(adj, x)

def one_step(edge_index, x):
    edge_weight = torch.ones(edge_index.size(1))
    if x>1: 
        adj = to_dense_adj(edge_index)[0]
        for k in range(2,x+1):
            adj_k = 1/k * (((adj_times_x(adj, k)>0)*1 - adj) > 0) * 1
            
            edge_index_k = (adj_k > 0).nonzero().t()
            edge_weight_k = 1/k * torch.ones(edge_index_k.size(1))
            edge_index = torch.cat((edge_index, edge_index_k), dim=1)
            edge_weight = torch.cat((edge_weight, edge_weight_k), dim=0)
            
    return edge_index, edge_weight