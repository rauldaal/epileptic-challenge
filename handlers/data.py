import math
from torch import Generator
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from objects import EpilepticDataset


def get_eliptic_dataloader(config):
    dataset = generate_eliptic_dataset(config)
    train, validation = train_test_splitter(dataset=dataset, split_value=0.8, seed=config.get("seed", 42))

    eliptic_data_loader_train = DataLoader(
        dataset=train,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
    )

    eliptic_data_loader_validation = DataLoader(
        dataset=validation,
        batch_size=config.get("batchSize"),
        shuffle=config.get("shufle"),
        num_workers=config.get("numWorkers"),
    )

    return eliptic_data_loader_train, eliptic_data_loader_validation, dataset.get_used_patients()


def generate_eliptic_dataset(config):
    eliptic_dataset = EpilepticDataset(
        parquet_folder=config.get("parquet_folder"),
        numpy_folder=config.get("numpy_folder"),
        transform=None
    ),
    return eliptic_dataset


def train_test_splitter(dataset, split_value, seed):
    size_train = math.ceil(len(dataset)*split_value)
    size_test = len(dataset)-size_train
    print(size_test, size_train)
    print(len(dataset))
    train, test = random_split(dataset, [size_train, size_test], generator=Generator().manual_seed(seed))
    return train, test
