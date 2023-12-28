import os
import json
import logging
import wandb
import uuid

from handlers import (
    generate_eliptic_dataset,
    generate_model_objects,
    map_configuration,
    load_model,
    perform_k_fold,
    save_model,
    )


def main(config):
    configurations = map_configuration(config_data=config)
    for config in configurations:
        if not config_data.get("execution_name"):
            config_data["executionName"] = config_data.get("projectName") + str(uuid.uuid4())[:-4]
        print(f"Configuration Parameters: {config}")
        with wandb.init(
            project=config.get("projectName"), name=config.get('execution_name'),
            notes='execution', tags=['main'],
            reinit = True, config=config):
            wandb.define_metric('train_loss', step_metric='epoch')
            wandb.define_metric('validation_loss', step_metric='epoch')

            if not config.get("model_name"):

                model, criterion, optimizer = generate_model_objects(config=config)
                dataset = generate_eliptic_dataset(config=config)
                perform_k_fold(config=config, model=model, criterion=criterion, optimizer=optimizer, dataset=dataset)

                save_model(model, config)
            else:
                model, criterion = load_model(config)


if __name__ == "__main__":
    with open("/fhome/mapsiv04/epileptic-challenge/config.json", "r") as f:
        config_data = json.load(f)
        main(config=config_data)


