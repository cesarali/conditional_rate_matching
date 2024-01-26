import os
import torch
from pathlib import Path
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.data.image_dataloader_config import DiscreteCIFAR10Config
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
import torchvision
from conditional_rate_matching.data.image_dataloader_config import DiscreteCIFAR10Config


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

class DiscreteCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, data_root,train=True,download=True,random_flips=False):
        super().__init__(root=data_root,
                         train=train,
                         download=download)

        self.data = torch.from_numpy(self.data)
        self.data = self.data.transpose(1,3)
        self.data = self.data.transpose(2,3)

        self.targets = torch.from_numpy(np.array(self.targets))

        # Put both data and targets on GPU in advance
        self.data = self.data.view(-1, 3, 32, 32)

        self.random_flips = random_flips
        if self.random_flips:
            self.flip = torchvision.transforms.RandomHorizontalFlip()

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'CIFAR10', 'processed')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.random_flips:
            img = self.flip(img)

        return img,target

class DiscreteCIFAR10Dataloader():
    """

    """
    def __init__(self,cfg:DiscreteCIFAR10Config,device=torch.device("cpu")):
        train_dataset = DiscreteCIFAR10(data_root=DiscreteCIFAR10Config.data_dir,train=True)
        test_dataset =  DiscreteCIFAR10(data_root=DiscreteCIFAR10Config.data_dir,train=False)


        self.number_of_spins = cfg.dimensions

        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=cfg.batch_size,
                                                            shuffle=True)

        self.test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                           batch_size=cfg.batch_size,
                                                           shuffle=True)
    def train(self):
        return self.train_dataloader

    def test(self):
        return self.test_dataloader

if __name__ =="__main__":
    from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

    data_config = NISTLoaderConfig(flatten=False,batch_size=23)
    dataloder,_ = get_data(data_config)
    databatch = next(dataloder.__iter__())


