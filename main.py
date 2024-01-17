import os
import json
import logging
import wandb
import uuid

from handlers import (
    generate_eliptic_dataset,
    generate_model_objects,
    get_eliptic_dataloader,
    map_configuration,
    load_model,
    perform_k_fold,
    save_model,
    test,
    compute_confussion_matrix
    )

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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


            train_dataset, test_dataset = generate_eliptic_dataset(config=config)
            if not config.get("model_name"):
                model_type="LSTM"
                model, criterion, optimizer = generate_model_objects(model_type,config=config)
                train_score, val_score = perform_k_fold(config=config, model=model, criterion=criterion, optimizer=optimizer, dataset=train_dataset)
                logging.info(train_score)
                logging.info(val_score)
                save_model(model, config)
                true_labels,pred_labels=test(model=model, criterion=criterion, test_data_loader=get_eliptic_dataloader(config=config, subset=test_dataset))
            else:
                print("test")
                model, criterion = load_model(config)
                true_labels,pred_labels=test(model=model, criterion=criterion, test_data_loader=get_eliptic_dataloader(config=config, subset=test_dataset))
                # print("true",true_labels)
                # print("predicted",pred_labels)
                compute_confussion_matrix(true=true_labels, pred=pred_labels, project_path=config.get("project_path"))


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config_data = json.load(f)
        main(config=config_data)


