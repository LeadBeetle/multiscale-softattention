import torch
torch.manual_seed(43)
from torch.nn import Linear as Lin
import torch.nn.functional as F
from tqdm import tqdm
from utils.utils import one_step, one_step_sparse

class Net(torch.nn.Module):
    __slots__ = ('device', 'num_layers', '_dropout', '_use_layer_norm', '_use_batch_norm', 'nbor_degree', 'adj_mode', 'sparse', 'skips')

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads, dropout, device, use_layer_norm=False, use_batch_norm=False, nbor_degree=1, adj_mode=None, sparse=True, computationBefore=True):
        super(Net, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self._dropout = dropout
        self._use_layer_norm = use_layer_norm
        self._use_batch_norm = use_batch_norm
        self.nbor_degree = nbor_degree
        self.adj_mode = adj_mode
        self.sparse = sparse
        self.computationBefore = computationBefore
        
        self.one_step_gen = one_step_sparse if sparse else one_step
        
        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))
        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            edge_weight = None
            if not self.computationBefore:
                edge_index, edge_weight = self.one_step_gen(edge_index, self.nbor_degree, x.size(0), self.device)
            if self.sparse:
                    edge_index = edge_index.t()
            x = self.convs[i]((x, x_target), edge_index, edge_weight)
            if self._use_layer_norm:
                x = self.layer_normalizations[i](x)
            if self._use_batch_norm:
                x = self.batch_normalizations[i](x)

            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self._dropout, training=self.training)
                              
        return x.log_softmax(dim=-1)

    def inference(self, x_all, loader):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to(self.device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(self.device)
                x_target = x[:size[1]]
                edge_weight = None
                if not self.computationBefore:
                    edge_index, edge_weight = self.one_step_gen(edge_index, self.nbor_degree, x.size(0), self.device)
                if self.sparse:
                    edge_index = edge_index.t()
                x = self.convs[i]((x, x_target), edge_index, edge_weight)
                if self._use_layer_norm:
                    x = self.layer_normalizations[i](x)
                if self._use_batch_norm:
                    x = self.batch_normalizations[i](x)

                x = x + self.skips[i](x_target)

                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

