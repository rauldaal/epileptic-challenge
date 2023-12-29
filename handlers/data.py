import math
from torch import Generator
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from objects import EpilepticDataset


def get_eliptic_dataloader(config, subset):

    eliptic_data_loader = DataLoader(
        dataset=subset,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
    )

    return eliptic_data_loader


def generate_eliptic_dataset(config):
    eliptic_dataset = EpilepticDataset(
        parquet_folder=config.get("parquet_folder"),
        numpy_folder=config.get("numpy_folder"),
        transform=None
    )
    train, test = train_test_splitter(dataset=eliptic_dataset, split_value=0.6, seed=config.get("seed"))
    return train, test


def train_test_splitter(dataset, split_value, seed):
    size_train = math.ceil(len(dataset)*split_value)
    size_test = len(dataset)-size_train
    print(size_test, size_train)
    print(len(dataset))
    train, test = random_split(dataset, [size_train, size_test], generator=Generator().manual_seed(seed))
    return train, test
