import torch
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms,datasets
from graph_bridges.data.transforms import SqueezeTransform
from graph_bridges.data.transforms import FlattenTransform
from graph_bridges.data.transforms import BinaryTensorToSpinsTransform

def get_data(dataset_name,config):
    data_= dataset_name

    batch_size = config.batch_size
    threshold = config.pepper_threshold
    dataloader_data_dir = config.data_dir

    transform = [transforms.ToTensor(),
                 transforms.Lambda(lambda x: (x > threshold).float())]

    if not config.as_image:
        transform.append(FlattenTransform)
        transform.append(SqueezeTransform)
    if config.as_spins:
        transform.append(BinaryTensorToSpinsTransform)

    transform = transforms.Compose(transform)

    # Load MNIST dataset
    if data_ == "mnist":
        train_dataset = datasets.MNIST(dataloader_data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(dataloader_data_dir, train=False, download=True, transform=transform)
    elif data_ == "emnist":
        train_dataset = datasets.EMNIST(root=dataloader_data_dir,
                                        split='letters',
                                        train=True,
                                        download=True,
                                        transform=transform)

        test_dataset = datasets.EMNIST(root=dataloader_data_dir,
                                       split='letters',
                                       train=False,
                                       download=True,
                                       transform=transform)
    elif data_== "fashion":
        train_dataset = datasets.FashionMNIST(root=dataloader_data_dir,
                                              train=True,
                                              download=True,
                                              transform=transform)
        test_dataset = datasets.FashionMNIST(root=dataloader_data_dir,
                                             train=False,
                                             download=True,
                                             transform=transform)
    else:
        raise Exception("Data Loader Not Found!")

    config.training_size = len(train_dataset)
    config.test_size = len(test_dataset)
    config.total_data_size = config.training_size + config.test_size

    config.test_split = config.test_size/config.total_data_size
    config.train_split = config.training_size/config.total_data_size

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=True)

    return train_loader,test_loader

class NISTLoader:

    name_ = "NISTLoader"

    def __init__(self, config,device):
        self.config = config

        self.batch_size = config.batch_size
        self.delete_data = config.delete_data
        self.number_of_spins = config.data.D

        self.dataloader_data_dir = config.data.dataloader_data_dir
        self.dataloader_data_dir_path = Path(self.dataloader_data_dir)
        self.dataloader_data_dir_file_path = Path(config.data.dataloader_data_dir_file)

        self.train_loader,self.test_loader = get_data(self.config)

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader

