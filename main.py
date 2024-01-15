import os
import json
import logging
import wandb
import uuid

from handlers import (
    generate_eliptic_dataset,
    generate_lstm_model_objects,
    generate_model_objects,
    get_eliptic_dataloader,
    map_configuration,
    load_model,
    perform_k_fold,
    save_model,
    test,
    test_lstm,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(config):
    configurations = map_configuration(config_data=config)
    for config in configurations:
        if not config_data.get("execution_name"):
            config_data["executionName"] = config_data.get("projectName") + str(uuid.uuid4())[:-4]
        logging.info(f"Configuration Parameters: {config}")
        with wandb.init(
            project=config.get("projectName"), name=config.get('execution_name'),
            notes='execution', tags=['main'],
            reinit = True, config=config):
            wandb.define_metric('train_loss', step_metric='epoch')
            wandb.define_metric('validation_loss', step_metric='epoch')
            wandb.define_metric('validation_loss_lstm', step_metric='epoch_lstm')
            wandb.define_metric('train_loss_lstm', step_metric='epoch_lstm')

            if not config.get("model_name"):

                model, criterion, optimizer = generate_model_objects(config=config)
                model_LSTM, criterion_LSTM, optimizer_LSTM = generate_lstm_model_objects(config=config)

                train_dataset, test_dataset = generate_eliptic_dataset(config=config)
                logging.info("KFOLD LSTM")
                train_score, val_score = perform_k_fold(config=config, model=model_LSTM, criterion=criterion_LSTM, optimizer=optimizer_LSTM, dataset=train_dataset, lstm=True)
                logging.info(train_score)
                logging.info(val_score)
                train_score, val_score = perform_k_fold(config=config, model=model, criterion=criterion, optimizer=optimizer, dataset=train_dataset)
                logging.info(train_score)
                logging.info(val_score)
                save_model(model, config)
                test_lstm(model=model_LSTM, criterion=criterion_LSTM, test_data_loader=get_eliptic_dataloader(config=config, subset=test_dataset))
                test(model=model, criterion=criterion, test_data_loader=get_eliptic_dataloader(config=config, subset=test_dataset))
                
                # Group kfold
                model, criterion, optimizer = generate_model_objects(config=config)
                model_LSTM, criterion_LSTM, optimizer_LSTM = generate_lstm_model_objects(config=config)
                # train_dataset, test_dataset = generate_eliptic_dataset(config=config)
                logging.info("GROUP KFOLD LSTM")
                train_score, val_score = perform_group_kfold(config=config, model=model, criterion=criterion, optimizer=optimizer, dataset=train_dataset)
                logging.info(train_score)
                logging.info(val_score)
                train_score, val_score = perform_group_kfold(config=config, model=model_LSTM, criterion=criterion_LSTM, optimizer=optimizer_LSTM, dataset=train_dataset, lstm=True)
                logging.info(train_score)
                logging.info(val_score)
                save_model(model, config)
                test_lstm(model=model_LSTM, criterion=criterion_LSTM, test_data_loader=get_eliptic_dataloader(config=config, subset=test_dataset))
                test(model=model, criterion=criterion, test_data_loader=get_eliptic_dataloader(config=config, subset=test_dataset))
                
            else:
                model, criterion = load_model(config)


if __name__ == "__main__":
    with open("C:/Users/Raul/OneDrive - UAB/4t/MA PSIV/RETO EPILIEPSIA/epileptic-challenge/config.json", "r") as f:
        config_data = json.load(f)
        main(config=config_data)


