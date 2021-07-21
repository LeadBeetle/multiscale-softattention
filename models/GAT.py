from models.Net import Net
import torch

from torch_geometric.nn import GATConv
from torch.nn import Linear as Lin
import torch.nn.functional as F
from tqdm import tqdm


class GAT(Net):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dataset, dropout, device, use_layer_norm=False):
        super(GAT, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 heads, dataset, dropout, device, use_layer_norm)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels,
                                  heads))
        
        self._layers_normalization = []
        if self._use_layer_norm:
            self._layers_normalization.append(torch.nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
            if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(hidden_channels)
                )
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))
        if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(out_channels)
                )
        self.layer_normalizations = torch.nn.ModuleList(self._layers_normalization)
        
