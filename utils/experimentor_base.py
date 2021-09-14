import os.path as osp

import torch
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F

from models.GAT import GAT
from models.GATv2 import GATV2
from models.Transformer import Transformer
import json
import datetime
import pprint
import logging
import sys
import time

import traceback
from utils.constants import * 



class Experimentor:
    
    def __init__(self, config):
        self.config = config
        
        pp = pprint.PrettyPrinter(indent=4)
        self.suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        filename = "_".join(["logs/my_log", self.suffix, ".txt"])
                
        logging.basicConfig(level=logging.DEBUG, filename=filename, filemode="a+",
                                        format="%(message)s")
        logging.info(pp.pformat(config))

        logging.info("\ntorch.cuda is available: " + str(torch.cuda.is_available()))
        
        
        self.dataset_name = config["dataset_name"]
        self.device = torch.device('cuda' if torch.cuda.is_available() and not self.config['force_cpu'] else 'cpu')
        
        logging.info(f"Used Dataset: {self.dataset_name}" )
        
        self.initData()
     

    def initData(self):
        root = osp.abspath( 'data')
        self.evaluator = Evaluator(name=datasetMapping[self.dataset_name])
        
        self.dataset = PygNodePropPredDataset(datasetMapping[self.dataset_name], root)
        self.split_idx = self.dataset.get_idx_split()
        
        self.data = self.dataset[0]
        self.train_idx = self.split_idx['train']
        self.val_idx = self.split_idx['valid']

        self.train_size = self.train_idx.size(0)
        
        self.criterion = F.nll_loss
        self.num_classes = self.dataset.num_classes
        self.num_features = self.dataset.num_features
        
        self.setLoaders()
        self.setModel()
          
    
    def setLoaders(self):
        self.x = self.data.x.to(self.device)
        self.y = self.data.y.squeeze().to(self.device)
        
        self.train_loader = NeighborSampler(self.data.edge_index, node_idx=self.train_idx,
                                    sizes=[10] * self.config["num_of_layers"], batch_size=self.config["batch_size"],
                                    shuffle=True, num_workers=self.config["num_workers"])
        self.test_loader = NeighborSampler(self.data.edge_index, node_idx=None, sizes=[-1],
                                        batch_size=self.config["test_batch_size"], shuffle=False,
                                        num_workers=self.config["num_workers"])  
    
    def setModel(self):
        if self.config["model_type"] == ModelType.GATV1:
            self.model = GAT(self.num_features, self.config["hidden_size"], self.num_classes, num_layers=self.config["num_of_layers"],
                heads=self.config["num_heads"], dropout = self.config["dropout"], device = self.device, use_layer_norm=self.config["use_layer_norm"], 
                use_batch_norm=self.config["use_layer_norm"], nbor_degree = self.config["nbor_degree"], adj_mode = self.config["adj_mode"], sparse = self.config["sparse"])
        elif self.config["model_type"] == ModelType.GATV2:
            self.model = GATV2(self.num_features, self.config["hidden_size"], self.num_classes, num_layers=self.config["num_of_layers"],
                heads=self.config["num_heads"], dropout = self.config["dropout"], device = self.device, use_layer_norm=self.config["use_layer_norm"], 
                use_batch_norm=self.config["use_layer_norm"], nbor_degree = self.config["nbor_degree"], adj_mode = self.config["adj_mode"], sparse = self.config["sparse"])  
        elif self.config["model_type"] == ModelType.TRANS:
            self.model = Transformer(self.num_features, self.config["hidden_size"], self.num_classes, num_layers=self.config["num_of_layers"],
                heads=self.config["num_heads"], dropout = self.config["dropout"], device = self.device, use_layer_norm=self.config["use_layer_norm"], 
                use_batch_norm=self.config["use_layer_norm"], nbor_degree = self.config["nbor_degree"], adj_mode = self.config["adj_mode"], sparse = self.config["sparse"])    
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
               
    def train(self, epoch):
        start = time.time()
        self.model.train()
        do_logging = epoch == 1 or self.config["do_train_tqdm_logging"]
        if do_logging:
            pbar = tqdm(total=self.train_size)
            pbar.set_description(f'Epoch {epoch:02d}')
        total_loss, total_correct = 0, 0
        for batch_size, n_id, adjs in self.train_loader:
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
        approx_acc = total_correct / self.train_size
        
        time_elapsed = time.time() - start
        return loss, approx_acc, time_elapsed


    
    @torch.no_grad()
    def test(self):
        start = time.time()
        self.model.eval()
        out = self.model.inference(self.x, loader = self.test_loader)

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

        time_elapsed = time.time() - start
        return train_acc, val_acc, test_acc, time_elapsed
    
    def run(self):
        test_freq = self.config['test_frequency'] or 10

        test_accs = []
        train_accs = []
        val_accs = []
        
        for run in range(1, 1 + self.config["num_of_runs"]):
            logging.info('')
            logging.info(f'Run {run:02d}:')
            print(f'Run {run:02d}:')
            logging.info('')

            self.model.reset_parameters()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

            best_val_acc, final_test_acc, final_train_acc, final_val_acc = 0, 0, 0, 0
            waited_iterations = 0
            for epoch in range(1, 1 + self.config["num_of_epochs"]):
                loss, acc, train_time = self.train(epoch)
                do_logging = epoch % self.config["console_log_freq"] == 0 or epoch == 1
                if do_logging:
                    logging.info(f'Epoch {epoch:02d}| Loss: {loss:.4f}| Acc: {acc:.4f}| Train Time: {train_time:.4f}s')
                    print(f'Epoch {epoch:02d}| Loss: {loss:.4f}| Acc: {acc:.4f}| Train Time: {train_time:.4f}s')

                if epoch % test_freq == 0:
                    train_acc, val_acc, test_acc, eval_time = self.test()
                    logging.info(f'Train: {train_acc:.4f}| Val: {val_acc:.4f}| '
                        f'Test: {test_acc:.4f}| Eval Time: {eval_time:.4f}s')
                    print(f'Train: {train_acc:.4f}| Val: {val_acc:.4f}| '
                        f'Test: {test_acc:.4f}| Eval Time: {eval_time:.4f}s')
                    
                    waited_iterations += test_freq
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        final_test_acc = test_acc
                        final_train_acc = train_acc
                        final_val_acc = val_acc
                        waited_iterations = 0
                    if waited_iterations >= self.config["patience_period"]:
                        break
                          
            print(f'\nResult of {run:2d}. run| Train: {final_train_acc:.4f}| Val: {final_val_acc:.4f}| Test: {final_test_acc:.4f}\n')
            test_accs.append(final_test_acc)
            train_accs.append(final_train_acc)
            val_accs.append(final_val_acc)

        test_acc = torch.tensor(test_accs)
        train_acc = torch.tensor(train_accs)
        val_acc = torch.tensor(val_accs)

        logging.info('\n============================')
        logging.info(f'Final Train: {train_acc.mean():.4f} ± {train_acc.std():.4f}')
        logging.info(f'Final Val: {val_acc.mean():.4f} ± {val_acc.std():.4f}')
        logging.info(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')

        data = self.config
             
        data["train_acc_mean"] = str(train_acc.mean().item())
        data["val_acc_mean"] = str(val_acc.mean().item())
        data["test_acc_mean"] = str(test_acc.mean().item())
        data["train_acc_std"] = str(train_acc.std().item())
        data["val_acc_std"] = str(val_acc.std().item())
        data["test_acc_std"] = str(test_acc.std().item())


        filename = "_".join(["results/res", self.suffix, ".json"])
        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def run_wrapper(self):
        try:
            self.run()
        except Exception as e:
            logging.info("Printing only the traceback above the current stack frame")
            logging.info("".join(traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2])))
            logging.info("\nPrinting the full traceback as if we had not caught it here...")
            logging.info(self.format_exception(e))

    def format_exception(self, e):
        exception_list = traceback.format_stack()
        exception_list = exception_list[:-2]
        exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
        exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

        exception_str = "Traceback (most recent call last):\n"
        exception_str += "".join(exception_list)
        # Removing the last \n
        exception_str = exception_str[:-1]

        return exception_str
        