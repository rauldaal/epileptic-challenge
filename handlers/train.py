import logging
import torch
from torch import nn
import wandb
import tqdm


#assert torch.cuda.is_available(), "GPU is not enabled"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_data_loader, validation_data_loader, optimizer, criterion, num_epochs, fold):
    for epoch in range(num_epochs):
        logging.info("+++++"*10)
        train_loss = 0

        model.train()
        for window, cls in tqdm.tqdm(train_data_loader):
            window = window.to(DEVICE, dtype=torch.float)
            cls = cls.to(DEVICE, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(window)
            loss = criterion(outputs, cls.view(-1, 1))
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_data_loader)
        logging.info("epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, num_epochs, train_loss))

        validation_loss = 0
        model.eval()
        with torch.no_grad():
            for window, cls in tqdm.tqdm(validation_data_loader):
                window = window.to(DEVICE, dtype=torch.float)
                cls = cls.to(DEVICE, dtype=torch.float)
                outputs = model(window)
                loss = criterion(outputs, cls.view(-1, 1))
                validation_loss += loss.item()
    
        validation_loss = validation_loss / len(validation_data_loader)
        logging.info("EPOCH : {}/{}, Validation Loss = {:.6f}".format(epoch + 1, num_epochs, validation_loss))
        wandb.log({"epoch": (fold*num_epochs+epoch), "train_loss": train_loss})
        wandb.log({"epoch": (fold*num_epochs+epoch), "validation_loss": validation_loss})

    return train_loss, validation_loss


def train_lstm(model, train_data_loader, validation_data_loader, optimizer, criterion, num_epochs, fold):
    for epoch in range(num_epochs):
        logging.info("+++++"*10)
        train_loss = 0
        model.train()
        for window, cls in tqdm.tqdm(train_data_loader):
            #window = torch.transpose(window, 0, 1)
            window = window.to(DEVICE, dtype=torch.float)
            batch = window.shape[0]
            cls = cls.to(DEVICE, dtype=torch.float)
            model.hidden_state = model.init_hidden(hidden_size=64, num_layers=8, batch_size=batch)
            optimizer.zero_grad()

            outputs = model(window)
            loss = criterion(outputs, cls.view(-1, 1))
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_data_loader)
        logging.info("epoch : {}/{}, Train loss = {:.6f}".format(epoch + 1, num_epochs, train_loss))

        validation_loss = 0
        model.eval()
        with torch.no_grad():
            for window, cls in tqdm.tqdm(validation_data_loader):
                window = window.to(DEVICE, dtype=torch.float)
                cls = cls.to(DEVICE, dtype=torch.float)
                batch = window.shape[0]
                model.hidden_state = model.init_hidden(hidden_size=64, num_layers=8, batch_size=batch)
                outputs = model(window)
                loss = criterion(outputs, cls.view(-1, 1))
                validation_loss += loss.item()
    
        validation_loss = validation_loss / len(validation_data_loader)
        logging.info("EPOCH : {}/{}, Validation Loss = {:.6f}".format(epoch + 1, num_epochs, validation_loss))
        wandb.log({"epoch_lstm": (fold*num_epochs+epoch), "train_loss_lstm": train_loss})
        wandb.log({"epoch_lstm": (fold*num_epochs+epoch), "validation_loss_lstm": validation_loss})
    return train_loss, validation_loss
    