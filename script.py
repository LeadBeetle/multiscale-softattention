# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from utils.experimentor_base import *
from utils.experimentor_proteins import * 
from utils.constants import *


# %%
config = {
    "dataset_name": Dataset.OGBN_PROTEINS,
    "model_type": ModelType.GATV1,
    'num_of_epochs': 1000,
    'num_of_runs': 3,
    'patience_period': 50,
    
    'batch_size': 1024,
    'test_batch_size': 1024,
    'num_workers': 2,
    'force_cpu': False,
    'test_frequency': 5,
    'console_log_freq': 1,
    'do_train_tqdm_logging': False,
    
    'lr': 0.01,
    'num_of_layers': 6, 
    'num_heads': 1,
    'hidden_size': 64,
    'dropout': 0.25,  
    "use_layer_norm": False,
    "use_batch_norm": True,
    
    'nbor_degree': 1,
    'adj_mode': AdjacencyMode.OneStep,
    'sparse': False
}


# %%
exp = Experimentor_Proteins if config["dataset_name"] == Dataset.OGBN_PROTEINS else Experimentor
experimentor = exp(config)
experimentor.run()


