# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from utils.experimentor import *
from utils.constants import *


# %%
config = {
    "dataset_name": Dataset.OGBN_PRODUCTS,
    "model_type": ModelType.GATV1,
    'num_of_epochs': 10000,
    'num_of_runs': 3,
    'patience_period': 200,
    
    'batch_size': 256,
    'test_batch_size': 256,
    'num_workers': 2,
    'force_cpu': False,
    'test_frequency': 5,
    'console_log_freq': 1,
    'do_train_tqdm_logging': False,
    
    'lr': 0.001,
    'num_of_layers': 3, 
    'num_heads': 1,
    'hidden_size': 128,
    'dropout': 0.5,  
    "use_layer_norm": False,
    
    'nbor_degree': 1,
    'adj_mode': AdjacencyMode.OneStep,
    'sparse': False
}


# %%
experimentor = Experimentor(config, logging)
experimentor.run()


