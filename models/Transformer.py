from models.base.Net import Net
import torch

from torch_geometric.nn import TransformerConv

class Transformer(Net):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dataset, dropout, device, use_layer_norm=False):
        super(Transformer, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 heads, dataset, dropout, device, use_layer_norm)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(dataset.num_features, hidden_channels,
                                  heads, dropout=dropout))
        
        self._layers_normalization = []
        if self._use_layer_norm:
            self._layers_normalization.append(torch.nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(heads * hidden_channels, hidden_channels, heads, dropout=dropout))
            if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(hidden_channels)
                )
        self.convs.append(
            TransformerConv(heads * hidden_channels, out_channels, heads,
                    concat=False, dropout=dropout))
        if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(out_channels)
                )
        self.layer_normalizations = torch.nn.ModuleList(self._layers_normalization)