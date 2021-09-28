import os.path as osp
from utils.experimentor_base import Experimentor

import torch
torch.manual_seed(43)
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import time
from ogb.nodeproppred import Evaluator

from utils.constants import * 

class Experimentor_Planetoid(Experimentor):
    def initData(self):
        root = osp.abspath('data')
        
        self.dataset = Planetoid(root, name=datasetMapping[self.dataset_name])
        self.evaluator = Evaluator(name="ogbn-arxiv")
        
        self.data = self.dataset[0]
        self.train_idx = self.data.train_mask
        self.val_idx = self.data.val_mask
        self.test_idx = self.data.test_mask
        self.num_nodes = self.data.num_nodes
        self.train_size = int(torch.sum(self.train_idx.int()))

        self.criterion = F.nll_loss
        self.num_classes = self.dataset.num_classes
        self.num_features = self.dataset.num_features
        
        self.setLoaders(ngb_size = -1)
        self.setModel()

    @torch.no_grad()
    def test(self):
        start = time.time()
        self.model.eval()
        out = self.model.inference(self.x, loader = self.test_loader)

        y_true = self.y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)

        train_acc = self.evaluator.eval({
            'y_true': y_true[self.train_idx],
            'y_pred': y_pred[self.train_idx],
        })['acc']
        val_acc = self.evaluator.eval({
            'y_true': y_true[self.val_idx],
            'y_pred': y_pred[self.val_idx],
        })['acc']
        test_acc = self.evaluator.eval({
            'y_true': y_true[self.test_idx],
            'y_pred': y_pred[self.test_idx],
        })['acc']

        time_elapsed = time.time() - start
        return train_acc, val_acc, test_acc, time_elapsed
