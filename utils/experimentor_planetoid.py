import os.path as osp
from utils.experimentor_base import Experimentor

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler
import time

from utils.constants import * 

class Experimentor_Planetoid(Experimentor):
    def initData(self):
        root = osp.abspath( 'data')
        
        self.dataset = Planetoid(root, name=datasetMapping[self.dataset_name])
        
        self.data = self.dataset[0]
        self.train_idx = self.data.train_mask
        self.val_idx = self.data.val_mask
        self.test_idx = self.data.test_mask
        
        self.train_size = self.train_idx.size(0) #int(torch.sum(self.train_idx.int()))

        self.criterion = F.nll_loss
        self.num_classes = self.dataset.num_classes
        self.num_features = self.dataset.num_features
        
        self.setLoaders()
        self.setModel()

    def setLoaders(self):
        self.x = self.data.x.to(self.device)
        self.y = self.data.y.squeeze().to(self.device)
        
        self.train_loader = NeighborSampler(self.data.edge_index, node_idx=None,
                                    sizes=[10] * self.config["num_of_layers"], batch_size=self.config["batch_size"],
                                    shuffle=True, num_workers=self.config["num_workers"])
        self.test_loader = NeighborSampler(self.data.edge_index, node_idx=None, sizes=[-1],
                                        batch_size=self.config["test_batch_size"], shuffle=False,
                                        num_workers=self.config["num_workers"])  

    @torch.no_grad()
    def test(self):
        start = time.time()
        self.model.eval()
        out = self.model.inference(self.x, loader = self.test_loader)

        y_true = self.y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = self.eval_acc(y_true[self.train_idx], y_pred[self.train_idx])['acc']
        val_acc = self.eval_acc(y_true[self.val_idx], y_pred[self.val_idx])['acc']
        test_acc = self.eval_acc(y_true[self.test_idx], y_pred[self.test_idx])['acc']

        time_elapsed = time.time() - start
        return train_acc, val_acc, test_acc, time_elapsed

    @torch.no_grad()
    def eval_acc(self, y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(torch.sum(correct.int()))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}
