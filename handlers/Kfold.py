import pandas as pd
from torch.utils.data import Subset

from .data import get_eliptic_dataloader

from .train import train

def get_fold_index(iter, num_folds, len_dataset):
    len_fraction = int(len_dataset/num_folds)
    idx_val_start = len_dataset * (iter * 1/num_folds)
    idx_val_stop = idx_val_start + len_fraction
    return idx_val_start, idx_val_stop


def perform_k_fold(config, model, criterion, optimizer, dataset):
    train_score = pd.Series()
    val_score = pd.Series()

    for i in range(config.get("num_folds")):
        idx_val_start, idx_val_stop = get_fold_index(i, config.get("num_folds"), len(dataset))
        idx_val = list(range(int(idx_val_start), int(idx_val_stop)))
        idx_train = list(set(range(len(dataset))) - set(idx_val))
        validation_subset = Subset(dataset, idx_val)
        train_subset = Subset(dataset, idx_train)
        train_loader = get_eliptic_dataloader(config, train_subset)
        validation_loader = get_eliptic_dataloader(config, validation_subset)
        train_acc, validation_acc = train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_data_loader=train_loader,
            validation_data_loader=validation_loader,
            num_epochs=config.get("num_epochs")
            )
        train_score.at[i] = train_acc
        val_score.at[i] = validation_acc

    return train_score, val_score