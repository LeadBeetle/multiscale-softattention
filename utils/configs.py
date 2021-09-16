from utils.constants import * 

base_config = {
    'num_of_epochs': 500,
    'num_of_runs': 1,
    'patience_period': 50,
    'num_workers': 0,
    'force_cpu': False,
    'test_frequency': 1,
    'console_log_freq': 1,
    'do_train_tqdm_logging': False,
    'adj_mode': AdjacencyMode.OneStep
}

coracite_config = {
    'batch_size': 256,
    'test_batch_size': 256,
    'lr': 0.005,
    'num_of_layers': 2, 
    'num_heads': 1,
    'hidden_size': 128,
    'dropout': 0.6,  
    "use_layer_norm": False,
    "use_batch_norm": False,
}

pub_config = {
    'batch_size': 256,
    'test_batch_size': 256,
    'lr': 0.01,
    'num_of_layers': 2, 
    'num_heads': 8,
    'hidden_size': 128,
    'dropout': 0.6,  
    "use_layer_norm": False,
    "use_batch_norm": False,
}

arxiv_config = {
    'batch_size': 20000,
    'test_batch_size': 20000,
    'lr': 0.01,
    'num_of_layers': 3, 
    'num_heads': 1,
    'hidden_size': 256,
    'dropout': 0.25,  
    "use_layer_norm": True,
    "use_batch_norm": False,
}

products_config = {
    'batch_size': 128,
    'test_batch_size': 64,
    'lr': 0.0001,
    'num_of_layers': 3, 
    'num_heads': 1,
    'hidden_size': 128,
    'dropout': 0.5  
}

proteins_config = {
    'batch_size': 64,
    'test_batch_size': 64,
    'lr': 0.01,
    'num_of_layers': 6, 
    'num_heads': 1,
    'hidden_size': 64,
    'dropout': 0.25  
}