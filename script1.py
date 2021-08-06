# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from utils.experimentor import *
from utils.constants import *


# %%
config = {
    "dataset_name": Dataset.OGBN_ARXIV,
    "model_type": ModelType.GATV1,
    'num_of_epochs': 10000,
    'num_of_runs': 10000,
    'patience_period': 200,
    
    #'should_visualize': False,
    'batch_size': 1024,
    'test_batch_size': 1024,
    'num_workers': 2,
    'force_cpu': False,
    'test_frequency': 10,
    'console_log_freq': 1,
    'do_train_tqdm_logging': False,
    
    'lr': 0.01,
    'num_of_layers': 2, 
    'num_heads': 1,
    'hidden_size': 256,
    'dropout': 0.25,  
    "use_layer_norm": True,
    
    'nbor_degree': 2,
    'adj_mode': AdjacencyMode.OneStep,
    'sparse': False
}


# %%
experimentor = Experimentor(config)
experimentor.run()


