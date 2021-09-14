from models.base.Net import Net
import torch

from models.convs.GatConv import GATConv


class GAT(Net):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout, device, use_layer_norm=False, use_batch_norm=False, nbor_degree=1, adj_mode=None, sparse=True, init_dropout=False):
        super(GAT, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout, device, use_layer_norm, use_batch_norm, nbor_degree, adj_mode, sparse, init_dropout)
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads, dropout=dropout))
        self._layers_normalization = []
        self._batch_normalization = []

        if self._use_layer_norm:
            self._layers_normalization.append(torch.nn.LayerNorm(hidden_channels))
        if self._use_batch_norm:
            self._batch_normalization.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers-2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads, dropout=dropout))
            if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(hidden_channels)
                )
            if self._use_batch_norm:
                self._batch_normalization.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False, dropout=dropout))
        if self._use_layer_norm:
                self._layers_normalization.append(
                    torch.nn.LayerNorm(out_channels)
                )
        if self._use_batch_norm:
            self._batch_normalization.append(torch.nn.BatchNorm1d(out_channels))
            
        self.layer_normalizations = torch.nn.ModuleList(self._layers_normalization)
        self.batch_normalizations = torch.nn.ModuleList(self._batch_normalization)
        
