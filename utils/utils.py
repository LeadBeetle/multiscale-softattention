import torch
from torch_geometric.utils import to_dense_adj


def edge_index_times_x(edge_index, x):
    adj = to_dense_adj(edge_index)[0]
    adj = torch.linalg.matrix_power(adj, x)
    edge_index = (adj > 0).nonzero().t()
    edge_weight = 1/x * torch.ones(edge_index.size(1))
          
    return edge_index, edge_weight