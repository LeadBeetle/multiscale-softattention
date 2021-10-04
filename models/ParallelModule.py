import torch
from torch_sparse.tensor import SparseTensor
from typing import Union
from torch_geometric.typing import (OptPairTensor)
from torch import Tensor
from utils.constants import * 

class ParallelModule(torch.nn.Sequential):
    def __init__(self, aggr_mode):
        super(ParallelModule, self).__init__( )
        self.aggr_mode = aggr_mode
        self.shared_weights = True

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, edge_weight=None):
        output = []
        for i, module in enumerate(self):
            cur_edge_index: SparseTensor = edge_index[i]
            output.append(module(x, cur_edge_index, edge_weight) )

        return self.aggregate(torch.stack(output, dim=0))

    def aggregate(self, input):
        if self.aggr_mode == AggrMode.MEAN:
            return self.mean(input)
        elif self.aggr_mode == AggrMode.MAX:
            return self.max(input)
        elif self.aggr_mode == AggrMode.MEDIAN:
            return self.median(input)
        elif self.aggr_mode == AggrMode.ADD:
            return self.add(input)

    def mean(self, input):
        return torch.mean(input, 0)

    def max(self, input):
        input_max, _ = torch.max(input, dim = 0)
        return input_max

    def median(self, input):
        input_median, _ = torch.median(input, 0)
        return input_median

    def add(self, input):
        return torch.sum(input, 0)

    def reset_parameters(self):
        for module in self: 
            module.reset_parameters()
