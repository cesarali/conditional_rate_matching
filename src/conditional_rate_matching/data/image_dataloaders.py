import torch
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms,datasets
from conditional_rate_matching.data.transforms import SqueezeTransform
from conditional_rate_matching.data.transforms import FlattenTransform
from conditional_rate_matching.data.transforms import CorrectEMNISTOrientation
from conditional_rate_matching.data.transforms import BinaryTensorToSpinsTransform
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from torch.utils.data import Subset

def get_data(config:NISTLoaderConfig):
    data_= config.dataset_name

    batch_size = config.batch_size
    threshold = config.pepper_threshold
    dataloader_data_dir = config.data_dir

    if data_ == "emnist":
        transform = [CorrectEMNISTOrientation(),
                     transforms.ToTensor(),
                     transforms.Lambda(lambda x: (x > threshold).float())]
    else:
        transform = [transforms.ToTensor(),
                     transforms.Lambda(lambda x: (x > threshold).float())]

    if config.flatten:
        transform.append(FlattenTransform)

    if not config.as_image:
        transform.append(SqueezeTransform)

    if config.unet_resize:
        transform.append(transforms.Resize((32, 32)))

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

    if config.max_training_size is not None:
        indices = list(range(config.max_training_size))
        train_dataset = Subset(train_dataset,indices)
    if config.max_test_size is not None:
        indices = list(range(config.max_test_size))
        test_dataset = Subset(test_dataset,indices)

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

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size

        self.train_loader,self.test_loader = get_data(self.config)
        self.dimensions = config.dimensions

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader

if __name__ =="__main__":
    from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

    data_config = NISTLoaderConfig(flatten=False,batch_size=23)
    dataloder,_ = get_data(data_config)
    databatch = next(dataloder.__iter__())
    print(databatch[0].shape)

