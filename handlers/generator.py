import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from objects import EpilepsModel, LSTMModel,EpilepsyLSTM


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_model_objects(config):
    model = EpilepsModel(**config).to(DEVICE)
    criterion = nn.BCELoss()
    if config.get("optimizer_type") == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate"))
    elif config.get("optimizer_type") == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=config.get("learning_rate"), rho=0.95, eps=1e-07)
    elif config.get("optimizer_type") == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config.get("learning_rate"),
            lr_decay=config.get("lerning_reate_decay"),
            weight_decay=config.get("weight_decay"),
            initial_accumulator_value=0.1, eps=1e-10
            )

    return model, criterion, optimizer
def get_default_hyperparameters():
   
    # initialize dictionaries
    inputmodule_params={}
    net_params={}
    outmodule_params={}
    
    # network input parameters
    inputmodule_params['n_nodes'] = 21
    
    # LSTM unit  parameters
    net_params['Lstacks'] = 1  # stacked layers (num_layers)
    net_params['dropout'] = 0.0
    net_params['hidden_size']= 256  #h
   
    # network output parameters
    outmodule_params['n_classes']=2
    outmodule_params['hd']=128
    
    return inputmodule_params, net_params, outmodule_params

def generate_lstm_model_objects(config):
        

    inputmodule_params, net_params, outmodule_params = get_default_hyperparameters()
    model = EpilepsyLSTM(inputmodule_params, net_params, outmodule_params).to(DEVICE)
    model.init_weights()
        
    criterion = nn.BCELoss()
    if config.get("optimizer_type") == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate"))
    elif config.get("optimizer_type") == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=config.get("learning_rate"), rho=0.95, eps=1e-07)
    elif config.get("optimizer_type") == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config.get("learning_rate"),
            lr_decay=config.get("lerning_reate_decay"),
            weight_decay=config.get("weight_decay"),
            initial_accumulator_value=0.1, eps=1e-10
            )

    return model, criterion, optimizer
def generate_lstm_model_objects_2(config):
    model = LSTMModel(**config).to(DEVICE)
    criterion = nn.BCELoss()
    if config.get("optimizer_type") == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate"))
    elif config.get("optimizer_type") == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=config.get("learning_rate"), rho=0.95, eps=1e-07)
    elif config.get("optimizer_type") == 'Adagrad':
        optimizer = optim.Adagrad(
            model.parameters(),
            lr=config.get("learning_rate"),
            lr_decay=config.get("lerning_reate_decay"),
            weight_decay=config.get("weight_decay"),
            initial_accumulator_value=0.1, eps=1e-10
            )

    return model, criterion, optimizer


def save_model(model, config):

    models_dir = os.path.join(config.get("project_path"), 'models')
    
    try:
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, config.get("execution_name") + '.pickle'), 'wb') as handle:
            pickle.dump(model, handle)
        logging.info("Modelo Guardado Correctamente")
    except Exception as e:
        logging.info(f"Error en el guardado. Error: {e}")


def load_model(config):

    models_dir = os.path.join(config.get("project_path"), 'models')
    with open(os.path.join(models_dir,config.get("execution_name")+".pickle"), 'rb') as handle:
        model = pickle.load(handle)
    criterion = nn.MSELoss()
    return model, criterion
