import torch
from torch_geometric.utils import to_dense_adj, degree


def one_step(edge_index, x):
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

    return edge_index, edge_weight


def one_step_sparse(edge_index, x):
    edge_weight = torch.ones(edge_index.size(1))
    e_size = edge_index.size(1)
    size = torch.Size([e_size, e_size] + list(edge_weight.size())[1:])
    adj = torch.sparse.FloatTensor(edge_index, edge_weight, size = size)
    if x>1: 
        adj_powers = [adj]
        for k in range(2, x+1):
            adj_powers.append(torch.sparse.mm(adj_powers[k-2], adj_powers[0]))
            adj_k = adj_powers[k-1] - adj_powers[0]
            adj_k = torch.sparse.FloatTensor(adj_k._indices(), 1/k * (adj_k._values()>0)*1, size = size)
            adj = adj + adj_k
            
    return adj