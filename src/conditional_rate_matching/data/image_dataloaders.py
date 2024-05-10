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

    name = "DiscreteCIFAR10"

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

#################
# Distorted MNIST
#################

from conditional_rate_matching.data.image_dataloader_config import DistortedNISTLoaderConfig
from torch.utils.data import DataLoader
from skimage.transform import swirl

def get_conditional_data(config: DistortedNISTLoaderConfig):
    data_= config.dataset_name
    threshold = config.pepper_threshold
    dataloader_data_dir = config.data_dir
    distortion = config.distortion
    distortion_level = config.distortion_level

    #...binerize MNIST images for dataset 1:

    transformation_list = [transforms.ToTensor(), transforms.Lambda(lambda x: (x > threshold).float())]

    #...define 1-parametric distortions for dataset 0:

    distortion_list=[]
    
    if distortion == 'noise': 
        distortion_list.append(transforms.ToTensor())
        distortion_list.append(transforms.Lambda(lambda x: add_gaussian_noise(x, std=distortion_level)))
    
    elif distortion == 'swirl': 
        distortion_list.append(transforms.Lambda(lambda x: apply_swirl(x, strength=distortion_level)))
        distortion_list.append(transforms.ToTensor())

    elif distortion == 'pixelate': 
        distortion_list.append(transforms.Lambda(lambda x: apply_coarse_grain(x, p=distortion_level)))
        distortion_list.append(transforms.ToTensor())

    elif distortion == 'half_mask':
        distortion_list.append(transforms.Lambda(lambda x: apply_half_mask(x)))
        distortion_list.append(transforms.ToTensor())

    distortion_list.append(transforms.Lambda(lambda x: (x > threshold).float()))

    #...reshape images accordingly:

    if config.flatten:
        transformation_list.append(FlattenTransform)
        distortion_list.append(FlattenTransform)

    if not config.as_image:
        transformation_list.append(SqueezeTransform)
        distortion_list.append(SqueezeTransform)

    if config.unet_resize:
        transformation_list.append(transforms.Resize((32, 32)))
        distortion_list.append(transforms.Resize((32, 32)))

    #...compose relevant transformations:
        
    distort = transforms.Compose(distortion_list)
    transform = transforms.Compose(transformation_list)

    # Load MNIST dataset

    if data_ == "mnist":
        train_data_0 = datasets.MNIST(dataloader_data_dir, train=True, download=True, transform=distort)
        test_data_0 = datasets.MNIST(dataloader_data_dir, train=False, download=True, transform=distort)
        train_data_1 = datasets.MNIST(dataloader_data_dir, train=True, download=True, transform=transform)
        test_data_1 = datasets.MNIST(dataloader_data_dir, train=False, download=True, transform=transform)
    else:
        raise Exception("Distortions only implemented for 'mnist' dataset!")

    return (train_data_0, test_data_0), (train_data_1, test_data_1)


class DistortedNISTLoaderDataEdge:
    def __init__(self, test_dl, train_dl):
        self.test_dl = test_dl
        self.train_dl = train_dl

    def train(self):
        return self.train_dl

    def test(self):
        return self.test_dl


class CoupledMNISTDataset(Dataset):
    def __init__(self, dataset_0, dataset_1):
        self.dataset_0 = dataset_0
        self.dataset_1 = dataset_1

    def __len__(self):
        return min(len(self.dataset_0), len(self.dataset_1))

    def __getitem__(self, idx):
        img_0, _ = self.dataset_0[idx]
        img_1, _ = self.dataset_1[idx]
        return img_0, img_1 

class DistortedNISTLoader:
    config: DistortedNISTLoaderConfig
    name: str = "DistortedNISTDataloader"

    def __init__(self, config: DistortedNISTLoaderConfig):
        """
        :param config:
        :param device:
        """
        self.config = config
        self.number_of_spins = self.config.dimensions
        data_0, data_1 = get_conditional_data(self.config)
        self.create_conditional_dataloaders(data_1, data_0)

    def train(self):
        return self.train_conditional()

    def test(self):
        return self.test_conditional()

    def train_conditional(self):
        for databatch in self.data_train:
            yield [databatch[0]],[databatch[1]]

    def test_conditional(self):
        for databatch in self.data_test:
            yield [databatch[0]],[databatch[1]]

    def define_sample_sizes(self):
        self.training_data_size = self.config.training_size
        self.test_data_size = self.config.test_size
        self.total_data_size = self.training_data_size + self.test_data_size
        self.config.training_proportion = float(self.training_data_size) / self.total_data_size

    def create_conditional_dataloaders(self, data_1, data_0):
        train_data_0, test_data_0 = data_0
        train_data_1, test_data_1 = data_1

        #=======================
        # INDEPENDENT
        #=======================

        self.data0_train = DataLoader(train_data_0, batch_size=self.config.batch_size, shuffle=True)
        self.data1_train = DataLoader(train_data_1, batch_size=self.config.batch_size, shuffle=True)
        self.data0_test = DataLoader(test_data_0, batch_size=self.config.batch_size, shuffle=True)
        self.data1_test = DataLoader(test_data_1, batch_size=self.config.batch_size, shuffle=True)
        self.data1 = DistortedNISTLoaderDataEdge(self.data1_test, self.data1_train)
        self.data0 = DistortedNISTLoaderDataEdge(self.data0_test, self.data0_train)

        #=======================
        # COUPLED
        #=======================

        train_ds = CoupledMNISTDataset(train_data_0, train_data_1)
        test_ds = CoupledMNISTDataset(test_data_0, test_data_1)
        self.data_train = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)
        self.data_test = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)

#...type of MNIST distortions

def add_gaussian_noise(tensor, mean=0., std=0.4):
    """Adds Gaussian noise to a tensor."""
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise

def apply_swirl(image, strength=5, radius=20):
    """Apply swirl distortion to an image."""
    image_np = np.array(image)
    swirled_image = swirl(image_np, strength=strength, radius=radius, mode='reflect')
    return Image.fromarray(swirled_image)

def apply_coarse_grain(image, p=0.7):
    """Coarse grains an image to a lower resolution."""
    old_size = image.size
    if p <= 0: return image  
    elif p >= 1: return Image.new('L', image.size, color=0)  # Return a black image
    new_size = max(1, int(image.width * (1 - p))), max(1, int(image.height * (1 - p)))
    image = image.resize(new_size, Image.BILINEAR)
    return image.resize(old_size, Image.NEAREST)  

def apply_half_mask(image):
    """ Masks the first half of the image along its width. """
    mask_height = int(image.height / 2)
    mask_width = image.width
    mask_size = (mask_width, mask_height)
    mask = Image.new('L', mask_size, color=255) 
    black_img = Image.new('L', image.size, color=0)
    black_img.paste(mask, (0, 0))  
    return Image.composite(image, black_img, black_img)

if __name__ =="__main__":
    from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

    data_config = NISTLoaderConfig(flatten=False,batch_size=23)
    dataloder,_ = get_data(data_config)
    databatch = next(dataloder.__iter__())


