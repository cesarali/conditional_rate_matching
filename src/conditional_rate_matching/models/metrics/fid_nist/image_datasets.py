import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from skimage.transform import swirl

from conditional_rate_matching import data_path

image_data_path = os.path.join(data_path,"raw")

def load_nist_data(name='MNIST', train=True, distortion=None, level=None):
    
    nist_datasets = ('MNIST', 'CIFAR10', 'CelebA', 'ImagNet', 'EMNIST Balanced', 'EMNIST Byclass', 'EMNIST Bymerge', 
                     'EMNIST Digits', 'EMNIST Letters', 'EMNIST mnist', 'QMNIST', 'KMNIST', 'FashionMNIST', 'USPS', 'SVHN', 'Omniglot',
                     'BinaryMNIST', 'BinaryCIFAR10', 'BinaryCelebA', 'BinaryImagNet', 'BinaryEMNIST Balanced', 'BinaryEMNIST Byclass', 
                     'BinaryEMNIST Bymerge', 'BinaryEMNIST Digits', 'BinaryEMNIST Letters', 'BinaryEMNIST mnist', 'BinaryQMNIST', 
                     'BinaryKMNIST', 'BinaryFashionMNIST', 'BinaryUSPS', 'BinarySVHN', 'BinaryOmniglot')

    assert name in nist_datasets, 'Dataset name not recognized. Choose between {}'.format(*nist_datasets)

    binerize_data = False
    if "Binary" in name: 
        binerize_data = True
        binary_threshold = {'BinaryMNIST': 0.5, 'BinaryFashionMNIST': 0.75, 'BinaryCIFAR10': 0.75, 'BinaryCelebA': 0.5, 'BinaryImagNet': 0.5, 
                            'BinaryEMNIST Balanced': 0.5, 'BinaryEMNIST Byclass': 0.5, 'BinaryEMNIST Bymerge': 0.5, 'BinaryEMNIST Digits': 0.5, 
                            'BinaryEMNIST Letters': 0.5, 'BinaryEMNIST mnist': 0.5, 'BinaryQMNIST': 0.5, 'BinaryKMNIST': 0.5, 'BinaryUSPS': 0.5, 
                            'BinarySVHN': 0.5, 'BinaryOmniglot': 0.5}

    transformation_list=[]
    
    #...define 1-parametric distortions:

    if distortion == 'noise': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: add_gaussian_noise(x,  mean=0., std=level)))
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x >  binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'blur':  
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.GaussianBlur(kernel_size=7, sigma=level))
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'swirl': 
        transformation_list.append(transforms.Lambda(lambda x: apply_swirl(x, strength=level, radius=20)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'pixelize': 
        transformation_list.append(transforms.Lambda(lambda x: apply_coarse_grain(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'crop': 
        transformation_list.append(transforms.Lambda(lambda x: apply_mask(x, p=level)))
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    elif distortion == 'binerize': 
        transformation_list.append(transforms.ToTensor())
        transformation_list.append(transforms.Lambda(lambda x: (x > level).type(torch.float32)))
        
    else:
        transformation_list.append(transforms.ToTensor())
        if binerize_data: 
            transformation_list.append(transforms.Lambda(lambda x: (x > binary_threshold[name]).type(torch.float32)))
    
    #...load dataset:
        
    if name == 'MNIST' or name == 'BinaryMNIST':
        return datasets.MNIST(root=image_data_path, train=train, download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('CIFAR10', 'BinaryCIFAR10'):
        return datasets.CIFAR10(root=image_data_path, train=train, download=True, transform=transforms.Compose(transformation_list))
    
    elif name in  ('CelebA', 'BinaryCelebA'):
        return datasets.CelebA(root=image_data_path, split='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('ImageNet', 'BinaryImageNet'):
        return datasets.ImageNet(root=image_data_path, split='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('EMNIST Balanced', 'BinaryEMNIST Balanced'):
        return datasets.EMNIST(root=image_data_path, split='balanced', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST Byclass', 'BinaryEMNIST Byclass'):
        return datasets.EMNIST(root=image_data_path, split='byclass', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST Bymerge', 'BinaryEMNIST Bymerge'):
        return datasets.EMNIST(root=image_data_path, split='bymerge', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST Digits', 'BinaryEMNIST Digits'):
        return datasets.EMNIST(root=image_data_path, split='digits', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST Letters', 'BinaryEMNIST Letters'):
        return datasets.EMNIST(root=image_data_path, split='letters', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('EMNIST mnist', 'BinaryEMNIST mnist'):
        return datasets.EMNIST(root=image_data_path, split='mnist', train=train, download=True, transform=transforms.Compose([CorrectEMNISTOrientation(), transforms.Compose(transformation_list)]))
    
    elif name in ('QMNIST', 'BinaryQMNIST'):
        return datasets.QMNIST(root=image_data_path, what='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('KMNIST', 'BinaryKMNIST'):
        return datasets.KMNIST(root=image_data_path, train=train, download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('FashionMNIST', 'BinaryFashionMNIST'):
        return datasets.FashionMNIST(root=image_data_path, train=train, download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('USPS', 'BinaryUSPS'):
        return datasets.USPS(root=image_data_path, train=train, download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('SVHN', 'BinarySVHN'):
        return datasets.SVHN(root=image_data_path, split='train', download=True, transform=transforms.Compose(transformation_list))
    
    elif name in ('Omniglot', 'BinaryOmniglot'):
        return datasets.Omniglot(root=image_data_path, download=True, transform=transforms.Compose(transformation_list))
    

class CorrectEMNISTOrientation(object):
    def __call__(self, img):
        return transforms.functional.rotate(img, -90).transpose(Image.FLIP_LEFT_RIGHT)


#...functions for applying perturbations to images:


def add_gaussian_noise(tensor, mean=0., std=1.):
    """Adds Gaussian noise to a tensor."""
    noise = torch.randn(tensor.size()) * std + mean
    return tensor + noise

def apply_swirl(image, strength=1, radius=20):
    """Apply swirl distortion to an image."""
    image_np = np.array(image)
    swirled_image = swirl(image_np, strength=strength, radius=radius, mode='reflect')
    return Image.fromarray(swirled_image)

def apply_coarse_grain(image, p=0.1):
    """Coarse grains an image to a lower resolution."""
    old_size = image.size
    if p <= 0: return image  
    elif p >= 1: return Image.new('L', image.size, color=0)  # Return a black image
    new_size = max(1, int(image.width * (1 - p))), max(1, int(image.height * (1 - p)))
    image = image.resize(new_size, Image.BILINEAR)
    return image.resize(old_size, Image.NEAREST)  # Resize back to 28x28

def apply_mask(image, p=0.1):
    """ Masks the image with a square window. """
    if p <= 0:
        return image  # No change
    elif p >= 1:
        return Image.new('L', image.size, color=0)  # Entirely black image

    mask_size = int(image.width * (1 - p)), int(image.height * (1 - p))
    mask = Image.new('L', mask_size, color=255)  # White square
    black_img = Image.new('L', image.size, color=0)
    black_img.paste(mask, (int((image.width - mask_size[0]) / 2), int((image.height - mask_size[1]) / 2)))
    return Image.composite(image, black_img, black_img)
