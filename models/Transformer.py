from models.base.Net import Net
import torch

from models.convs.TransformerConv import TransformerConv

class Transformer(Net):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout, device, use_layer_norm=False, use_batch_norm=False, nbor_degree=1, adj_mode=None, sparse=True, init_dropout=False):
        super(Transformer, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout, device, use_layer_norm, use_batch_norm, nbor_degree, adj_mode, sparse, init_dropout)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, hidden_channels,
                                  heads, dropout=dropout))
        
        self._layers_normalization = []
        if self._use_layer_norm:
            self._layers_normalization.append(torch.nn.LayerNorm(hidden_channels))
        
        for _ in range(num_layers-2):
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