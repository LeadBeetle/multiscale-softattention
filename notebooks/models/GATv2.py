from models.base.Net import Net
import torch
torch.manual_seed(43)

from models.convs.GATv2Conv import GATv2Conv
from utils.constants import *
from models.ParallelModule import *

class GATV2(Net):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout, device, use_layer_norm=False, use_batch_norm=False, 
                 nbor_degree=1, adj_mode=None, sparse=True, computationBefore=False, aggr_mode = AggrMode.MEAN):
        super(GATV2, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout, device, use_layer_norm, use_batch_norm, nbor_degree, adj_mode, sparse, computationBefore, aggr_mode)
        
        self.convs = torch.nn.ModuleList()
        self.add_conv(GATv2Conv(in_channels, hidden_channels,
                                  heads, dropout=dropout))
        
        self._layers_normalization = []
        if self._use_layer_norm:
            self._layers_normalization.append(torch.nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers-2):
            self.add_conv(GATv2Conv(heads * hidden_channels, hidden_channels, heads, dropout=dropout))
            if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(hidden_channels)
                )
        self.add_conv(GATv2Conv(heads * hidden_channels, out_channels, heads,
                    concat=False, dropout=dropout))
        if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(out_channels)
                )
        self.layer_normalizations = torch.nn.ModuleList(self._layers_normalization)