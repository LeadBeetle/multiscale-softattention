import os.path as osp
from utils.experimentor_base import Experimentor

import torch
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
import datetime
import logging
from torch_geometric.datasets import Planetoid
import time
import numpy as np

from utils.constants import * 

suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
filename = "_".join(["logs/my_log", suffix, ".txt"])
        
logging.basicConfig(level=logging.DEBUG, filename=filename, filemode="a+",
                                format="%(message)s")

class Experimentor_Planetoid(Experimentor):
    def initData(self):
        root = osp.abspath( 'data')
        
        self.dataset = Planetoid(root, name=datasetMapping[self.dataset_name])
        
        self.data = self.dataset[0]
        self.train_idx = self.data.train_mask
        self.val_idx = self.data.val_mask
        self.test_idx = self.data.test_mask
        
        self.train_size = int(torch.sum(self.train_idx.int()))

        self.criterion = F.nll_loss
        self.num_classes = self.dataset.num_classes
        self.num_features = self.dataset.num_features
        
        self.setLoaders()
        self.setModel()

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


    def eval_acc(self, y_true, y_pred):
        acc_list = []
        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(torch.sum(correct.int()))/len(correct))

        return {'acc': sum(acc_list)/len(acc_list)}
