import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from torch.utils.data import Subset, Dataset

from .data import get_eliptic_dataloader

from .train import train, train_lstm

def get_fold_index(iter, num_folds, len_dataset):
    len_fraction = int(len_dataset/num_folds)
    idx_val_start = len_dataset * (iter * 1/num_folds)
    idx_val_stop = idx_val_start + len_fraction
    return idx_val_start, idx_val_stop


def perform_k_fold(config, model, criterion, optimizer, dataset, lstm=False):
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
        if not lstm:
            logging.info("NORMAL TRAIN")
            train_acc, validation_acc = train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_data_loader=train_loader,
                validation_data_loader=validation_loader,
                num_epochs=config.get("num_epochs"),
                fold=i
                )
        else:
            logging.info("LSTM TRAIN")
            train_acc, validation_acc = train_lstm(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_data_loader=train_loader,
                validation_data_loader=validation_loader,
                num_epochs=config.get("num_epochs"),
                fold=i
                )
        train_score.at[i] = train_acc
        val_score.at[i] = validation_acc
        logging.info(f"***********FOLD {i} FINISED**********")
    return train_score, val_score

def perform_group_kfold(config, model, criterion, optimizer, dataset, lstm=False):
    train_score = pd.Series()
    val_score = pd.Series()
    custom_kfold = CustomKFold(n_splits=config.get("num_folds"), shuffle=True, random_state=42)
    X = dataset.dataset.data.iloc[dataset.indices]
    groups = X['filename'].values
    print(len(groups))
    i = 0
    for idx_train, idx_val in custom_kfold.split(dataset, groups=groups):
        logging.info(f"Train indices: {idx_train}")
        logging.info(f"Train groups: {groups[idx_train]}")
        logging.info(f"val indices: {idx_val}")
        logging.info(f"val groups: {groups[idx_val]}")
        print(np.max(idx_train))
        print(np.max(idx_val))
        train_subset = Subset(dataset, idx_train)
        validation_subset = Subset(dataset, idx_val)
        
        train_loader = get_eliptic_dataloader(config, train_subset)
        validation_loader = get_eliptic_dataloader(config, validation_subset)
        if lstm:
            train_acc, validation_acc = train_lstm(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_data_loader=train_loader,
                validation_data_loader=validation_loader,
                num_epochs=config.get("num_epochs"),
                fold=i
                )
        else:
            train_acc, validation_acc = train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_data_loader=train_loader,
                validation_data_loader=validation_loader,
                num_epochs=config.get("num_epochs"),
                fold=i
                )
        train_score.at[i] = train_acc
        val_score.at[i] = validation_acc
        i += 1

    return train_score, val_score
    
# Convert the DataFrame to a PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Implement your data retrieval logic here
        # For example, if you have a column 'features' and 'target' in your DataFrame:
        features = torch.tensor(self.data.iloc[index]['features'])
        target = torch.tensor(self.data.iloc[index]['target'])
        return features, target

class CustomKFold(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        # n_samples = X.shape[0]
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(indices)

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size

            test_indices = indices[start:end]

            group_mask = np.isin(groups[test_indices], groups)
            while not np.all(group_mask):
                test_indices = indices[start:end]
                group_mask = np.isin(groups[test_indices], groups)

            yield test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
