import os.path as osp

import torch 
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np

from models.GAT import GAT
from models.GATv2 import GATV2
from models.Transformer import Transformer
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.utils import to_undirected

from utils.constants import * 

class Experimentor:
    
    def __init__(self, config):
        print("torch.cuda is available:", torch.cuda.is_available())
        self.config = config
        self.dataset_name = config["dataset_name"]
        print("Used Dataset:", self.dataset_name)
        self.initData()
     
    def initData(self):
        root = osp.abspath( 'data')
        
        self.dataset = PygNodePropPredDataset(datasetMapping[self.dataset_name], root)
        self.split_idx = self.dataset.get_idx_split()
        self.evaluator = Evaluator(name=datasetMapping[self.dataset_name])
        self.data = self.dataset[0]
        self.train_idx = self.split_idx['train']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and not self.config['force_cpu'] else 'cpu')
        self.criterion = F.nll_loss
        
        self.setLoaders()
        self.setModel()
          
    
    def setLoaders(self):
        edge_index = self.data.edge_index
        if self.dataset_name == Dataset.OGBN_PROTEINS:
            ##Currently not supported since multi-task dataset
            _, col = self.data.edge_index
            self.data.x = scatter(self.data.edge_attr, col, 0, dim_size=self.data.num_nodes, reduce='add')
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.dataset_name == Dataset.OGBN_MAG:
            edge_index_dict = self.data.edge_index_dict
            # We need to add reverse edges to the heterogeneous graph.
            r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
            edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

            r, c = edge_index_dict[('author', 'writes', 'paper')]
            edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

            r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
            edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

            # Convert to undirected paper <-> paper relation.
            edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
            edge_index_dict[('paper', 'cites', 'paper')] = edge_index
            out = group_hetero_graph(edge_index_dict, self.data.num_nodes_dict)
            edge_index, edge_type, node_type, idx, local2global, key2int = out
            self.train_idx = local2global['paper'][self.split_idx['train']['paper']]
            #self.train_idx = self.split_idx['train']["paper"]
            self.data.x = self.data.x_dict["paper"]
            self.data.y = self.data.y_dict["paper"]
            
        self.train_loader = NeighborSampler(edge_index, node_idx=self.train_idx,
                                    sizes=[10] * self.config["num_of_layers"], batch_size=self.config["batch_size"],
                                    shuffle=True, num_workers=self.config["num_workers"])
        self.subgraph_loader = NeighborSampler(edge_index, node_idx=None, sizes=[-1],
                                        batch_size=self.config["test_batch_size"], shuffle=False,
                                        num_workers=self.config["num_workers"])  
        
        self.x = self.data.x.to(self.device)
        self.y = self.data.y.squeeze().to(self.device)
    
    def setModel(self):
        if self.config["model_type"] == ModelType.GATV1:
            self.model = GAT(self.dataset.num_features, self.config["hidden_size"], self.dataset.num_classes, num_layers=self.config["num_of_layers"],
                heads=self.config["num_heads"], dataset = self.dataset, dropout = self.config["dropout"], device = self.device, use_layer_norm=self.config["use_layer_norm"])
        elif self.config["model_type"] == ModelType.GATV2:
            self.model = GATV2(self.dataset.num_features, self.config["hidden_size"], self.dataset.num_classes, num_layers=self.config["num_of_layers"],
                heads=self.config["num_heads"], dataset = self.dataset, dropout = self.config["dropout"], device = self.device)  
        elif self.config["model_type"] == ModelType.TRANS:
            self.model = Transformer(self.dataset.num_features, self.config["hidden_size"], self.dataset.num_classes, num_layers=self.config["num_of_layers"],
                heads=self.config["num_heads"], dataset = self.dataset, dropout = self.config["dropout"], device = self.device)    
        
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
               
    def train(self, epoch):
        self.model.train()
        do_logging = epoch == 1 or self.config["do_train_tqdm_logging"]
        if do_logging:
            pbar = tqdm(total=self.train_idx.size(0))
            pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = total_correct = 0
        for batch_size, n_id, adjs in self.train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(self.device) for adj in adjs]
            self.optimizer.zero_grad()
            out = self.model(self.x[n_id], adjs)
            loss = self.criterion(out, self.y[n_id[:batch_size]])
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(self.y[n_id[:batch_size]]).sum())
            if do_logging:
                pbar.update(batch_size)

        if do_logging:
            pbar.close()

        loss = total_loss / len(self.train_loader)
        approx_acc = total_correct / self.train_idx.size(0)

        return loss, approx_acc

    @torch.no_grad()
    def test(self):
        self.model.eval()

        out = self.model.inference(self.x, loader = self.subgraph_loader)

        y_true = self.y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = self.evaluator.eval({
            'y_true': y_true[self.split_idx['train']],
            'y_pred': y_pred[self.split_idx['train']],
        })['acc']
        val_acc = self.evaluator.eval({
            'y_true': y_true[self.split_idx['valid']],
            'y_pred': y_pred[self.split_idx['valid']],
        })['acc']
        test_acc = self.evaluator.eval({
            'y_true': y_true[self.split_idx['test']],
            'y_pred': y_pred[self.split_idx['test']],
        })['acc']

        return train_acc, val_acc, test_acc
    
    def run(self):
        test_freq = self.config['test_frequency'] or 10

        test_accs = []
        for run in range(1, 1 + self.config["num_of_runs"]):
            print('')
            print(f'Run {run:02d}:')
            print('')

            self.model.reset_parameters()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

            best_val_acc = final_test_acc = 0
            waited_iterations = 0
            for epoch in range(1, 1 + self.config["num_of_epochs"]):
                loss, acc = self.train(epoch)
                do_logging = epoch % self.config["console_log_freq"] == 0 or epoch == 1
                if do_logging:
                    print(f'Epoch {epoch:02d}| Loss: {loss:.4f}, Approx. Train: {acc:.4f}')

                if epoch % test_freq == 0:
                    train_acc, val_acc, test_acc = self.test()
                    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
                        f'Test: {test_acc:.4f}')
                    waited_iterations += test_freq
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                        waited_iterations = 0
                    if waited_iterations >= self.config["patience_period"]:
                        break
                        
            test_accs.append(final_test_acc)
        test_acc = torch.tensor(test_accs)
        print('============================')
        print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')