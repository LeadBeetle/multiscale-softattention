from utils.constants import * 

base_config = {
    'num_of_epochs': 500,
    'num_of_runs': 1,
    'patience_period': 30,
    'num_workers': 0,
    'force_cpu': False,
    'test_frequency': 1,
    'console_log_freq': 1,
    'do_train_tqdm_logging': False,
    'adj_mode': AdjacencyMode.OneStep,
    'computationBefore': False
}

coracite_config = {
    'batch_size': 256,
    'test_batch_size': 256,
    'lr': 0.005,
    #'num_of_layers': 3, 
    'num_heads': 4,
    'hidden_size': 64,
    'dropout': 0.6,  
    "use_layer_norm": False,
    "use_batch_norm": False,
}

cora_config = {
    'force_cpu': True,
}

pub_config = {
    'batch_size': 256,
    'test_batch_size': 256,
    'lr': 0.01,
    #'num_of_layers': 3, 
    'num_heads': 4,
    'hidden_size': 128,
    'dropout': 0.6,  
    "use_layer_norm": False,
    "use_batch_norm": False,
}

arxiv_config = {
    'batch_size': 20000,
    'test_batch_size': 20000,
    'lr': 0.01,
    #'num_of_layers': 3, 
    'num_heads': 1,
    'hidden_size': 256,
    'dropout': 0.25,  
    "use_layer_norm": True,
    "use_batch_norm": False,
    'test_frequency': 5,
    'num_of_runs': 3,
    'computationBefore': False
}

products_config = {
    'batch_size': 128,
    'test_batch_size': 64,
    'lr': 0.001,
    #'num_of_layers': 3, 
    'num_heads': 1,
    'hidden_size': 128,
    'dropout': 0.5,
    "use_layer_norm": False,
    "use_batch_norm": False,
    'test_frequency': 10,
    'num_of_runs': 1,
    'computationBefore': False
}

proteins_config = {
    'batch_size': 64,
    'test_batch_size': 64,
    'lr': 0.01,
    #'num_of_layers': 3, 
    'num_heads': 8,
    'hidden_size': 64,
    'dropout': 0.25,
    "use_layer_norm": False,
    "use_batch_norm": False,
    'test_frequency': 10, 
    'num_of_runs': 1,
    'computationBefore': False
}

arx_trans_config = {
    'lr': 0.001,
    'dropout': 0.3,
    'hidden_size': 128, 
    'num_heads' : 4,
    #'num_of_layers': 3,
    'num_of_runs': 1
}

prod_trans_config = {
    'lr': 0.001,
    'dropout': 0.3,
    'hidden_size': 128,
    'num_heads' : 4,
    #'num_of_layers': 3,
    'num_of_runs': 1
}

proteins_trans_config = {
    'lr': 0.001,
    'dropout': 0.1,
    'hidden_size': 64,
    'num_heads' : 4,
    #'num_of_layers': 4,
    'num_of_runs': 1
}
