import os.path as osp
from utils.experimentor_base import Experimentor

import torch
from tqdm import tqdm
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.utils import to_undirected
import json
import datetime
import logging
import time

from utils.constants import * 

suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "_".join(["logs/my_log", suffix, ".txt"])
        
logging.basicConfig(level=logging.DEBUG, filename=filename, filemode="a+",
                                format="%(message)s")

class Experimentor_Proteins(Experimentor):

    def setLoaders(self):
        self.setProteinsData()
        
        self.train_loader = NeighborSampler(self.data.edge_index, node_idx=self.train_idx,
                                    sizes=[10] * self.config["num_of_layers"], batch_size=self.config["batch_size"],
                                    shuffle=True, num_workers=self.config["num_workers"])
        self.eval_loader = NeighborSampler(self.data.edge_index, node_idx=self.val_idx, sizes=[-1],
                                    batch_size=self.config["test_batch_size"], shuffle=False,
                                    num_workers=self.config["num_workers"])  
        self.test_loader = NeighborSampler(self.data.edge_index, node_idx=None, sizes=[-1],
                                        batch_size=self.config["test_batch_size"], shuffle=False,
                                        num_workers=self.config["num_workers"])  
            
    def setProteinsData(self):   
        _, col = self.data.edge_index
        self.x = scatter(self.data.edge_attr, col, 0, dim_size=self.data.num_nodes, reduce='add').to(self.device)
        self.y = self.data.y.squeeze().to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.num_classes = 112 # Num_Classes
        self.num_features = 8

             
    def train(self, epoch):
        start = time.time()
        self.model.train()
        do_logging = epoch == 1 or self.config["do_train_tqdm_logging"]
        if do_logging:
            pbar = tqdm(total=self.train_idx.size(0))
            pbar.set_description(f'Epoch {epoch:02d}')
        total_loss = 0
        for batch_size, n_id, adjs in self.train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(self.device) for adj in adjs]
            self.optimizer.zero_grad()
            out = self.model(self.x[n_id], adjs)
            loss = self.criterion(out, self.y[n_id[:batch_size]].to(torch.float32))
            loss.backward()
            self.optimizer.step()
            
            total_loss += float(loss)

            if do_logging:
                pbar.update(batch_size)

        if do_logging:
            pbar.close()

        loss = total_loss / len(self.train_loader)

        time_elapsed = time.time() - start
        return loss, None, time_elapsed

    @torch.no_grad()
    def test(self):
        start = time.time()
        self.model.eval()

        out = self.model.inference(self.x, loader = self.test_loader)

        y_true = self.y.cpu()
        y_pred = out
        
        train_roc = self.evaluator.eval({
            'y_true': y_true[self.split_idx['train']],
            'y_pred': y_pred[self.split_idx['train']],
        })['rocauc']
        val_roc = self.evaluator.eval({
            'y_true': y_true[self.split_idx['valid']],
            'y_pred': y_pred[self.split_idx['valid']],
        })['rocauc']
        test_roc = self.evaluator.eval({
            'y_true': y_true[self.split_idx['test']],
            'y_pred': y_pred[self.split_idx['test']],
        })['rocauc']

        time_elapsed = time.time() - start
        return train_roc, val_roc, test_roc, time_elapsed
    
    def run(self):
        test_freq = self.config['test_frequency'] or 10

        test_rocs = []
        train_rocs = []
        val_rocs = []
        for run in range(1, 1 + self.config["num_of_runs"]):
            logging.info('')
            logging.info(f'Run {run:02d}:')
            print(f'Run {run:02d}:')
            logging.info('')

            self.model.reset_parameters()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

            best_val_roc, final_test_roc, final_train_roc, final_val_roc = 0, 0, 0, 0
            waited_iterations = 0
            for epoch in range(1, 1 + self.config["num_of_epochs"]):
                loss, _, train_time = self.train(epoch)
                do_logging = epoch % self.config["console_log_freq"] == 0 or epoch == 1
                if do_logging:
                    logging.info(f'Epoch {epoch:02d}| Loss: {loss:.4f}| Train Time: {train_time:.4f}s')
                    print(f'Epoch {epoch:02d}| Loss: {loss:.4f}| Train Time: {train_time:.4f}s')

                if epoch % test_freq == 0:
                    train_roc, val_roc, test_roc, eval_time = self.test()
                    logging.info(f'Train: {train_roc:.4f}| Val: {val_roc:.4f}| '
                        f'Test: {test_roc:.4f}| Eval Time: {eval_time:.4f}s')
                    print(f'Train: {train_roc:.4f}| Val: {val_roc:.4f}| '
                        f'Test: {test_roc:.4f}| Eval Time: {eval_time:.4f}s')
                    waited_iterations += test_freq
                    if val_roc > best_val_roc:
                        best_val_roc = val_roc
                        final_test_roc = test_roc
                        final_train_roc = train_roc
                        final_val_roc = val_roc
                        waited_iterations = 0
                    if waited_iterations >= self.config["patience_period"]:
                        break

            print(f'\nResult of {run:2d}. run| Train: {final_train_roc:.4f}| Val: {final_val_roc:.4f}| Test: {final_test_roc:.4f}\n')
            test_rocs.append(final_test_roc)
            train_rocs.append(final_train_roc)
            val_rocs.append(final_val_roc)

        test_roc = torch.tensor(test_rocs)
        train_roc = torch.tensor(train_rocs)
        val_roc = torch.tensor(val_rocs)

        logging.info('\n============================')
        logging.info(f'Final Train: {train_roc.mean():.4f} ± {train_roc.std():.4f}')
        logging.info(f'Final Val: {val_roc.mean():.4f} ± {val_roc.std():.4f}')
        logging.info(f'Final Test: {test_roc.mean():.4f} ± {test_roc.std():.4f}')

        data = self.config
             
        data["train_roc_mean"] = str(train_roc.mean().item())
        data["val_roc_mean"] = str(val_roc.mean().item())
        data["test_roc_mean"] = str(test_roc.mean().item())
        data["train_roc_std"] = str(train_roc.std().item())
        data["val_roc_std"] = str(val_roc.std().item())
        data["test_roc_std"] = str(test_roc.std().item())


        filename = "_".join(["results/res", suffix, ".json"])
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        