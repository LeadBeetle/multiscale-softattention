from models.base.Net import Net
import torch

from models.convs.GATv2Conv import GATv2Conv

class GATV2(Net):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dataset, dropout, device, use_layer_norm=False, nbor_degree=1, adj_mode=None, sparse=True):
        super(GATV2, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 heads, dataset, dropout, device, use_layer_norm, nbor_degree, adj_mode, sparse)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(dataset.num_features, hidden_channels,
                                  heads, dropout=dropout))
        
        self._layers_normalization = []
        if self._use_layer_norm:
            self._layers_normalization.append(torch.nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(heads * hidden_channels, hidden_channels, heads, dropout=dropout))
            if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(hidden_channels)
                )
        self.convs.append(
            GATv2Conv(heads * hidden_channels, out_channels, heads,
                    concat=False, dropout=dropout))
        if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(out_channels)
                )
        self.layer_normalizations = torch.nn.ModuleList(self._layers_normalization)