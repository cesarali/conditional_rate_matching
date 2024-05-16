import torch
from pprint import pprint
from dataclasses import dataclass,asdict
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

def sample_categorical_from_dirichlet(config:StatesDataloaderConfig,return_tensor_samples=False):
    """

    :param probs:
    :param alpha:
    :param sample_size:
    :param dimensions:
    :param vocab_size:
    :param test_split:
    :return:
    """

    probs = config.bernoulli_probability
    alpha = config.dirichlet_alpha
    sample_size = config.sample_size
    dimensions = config.dimensions
    vocab_size = config.vocab_size
    test_split = config.test_split
    batch_size = config.batch_size
    max_test_size = config.max_test_size

    # ensure we have the probabilites
    if probs is None:
        if isinstance(alpha, float):
            alpha = torch.full((vocab_size,), alpha)
        else:
            assert len(alpha.shape) == 1
            assert alpha.size(0) == vocab_size
        # Sample from the Dirichlet distribution
        probs = torch.distributions.Dirichlet(alpha).sample([dimensions])
    else:
        if isinstance(probs,(np.ndarray,list)):
            probs = torch.Tensor(probs)
        probs = probs.squeeze()
        assert probs.max() <= 1.
        assert probs.max() >= 0.
        
    # Sample from the categorical distribution using the Dirichlet samples as probabilities
    distribution_per_dimension = Categorical(probs)
    categorical_samples = distribution_per_dimension.sample([sample_size]).float()

    test_size = int(test_split * categorical_samples.size(0))
    train_samples = categorical_samples[test_size:]
    test_samples = categorical_samples[:test_size]

    if max_test_size is not None:
        if test_samples.size(0) > max_test_size:
            test_samples = test_samples[:max_test_size]

    train_dataset, test_dataset = TensorDataset(train_samples), TensorDataset(test_samples)

    training_data_size = len(train_dataset)
    test_data_size = len(test_dataset)
    total_data_size = training_data_size + test_data_size

    config.training_size = training_data_size
    config.test_size = test_data_size
    config.total_data_size = total_data_size
    config.training_proportion = float(training_data_size) / total_data_size

    if return_tensor_samples:
        return train_samples,test_samples,probs
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size, shuffle=True)

        return train_loader, test_loader,probs


class StatesDataloader:

    name_ = "StatesDataloader"

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size

        self.train_loader,self.test_loader,self.probs = sample_categorical_from_dirichlet(self.config)
        self.config.bernoulli_probability = self.probs.numpy().tolist()
        self.dimensions = config.dimensions

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader


def set_probabilities(config:StatesDataloaderConfig,return_tensor_samples=False):
    """

    :param probs:
    :param alpha:
    :param sample_size:
    :param dimensions:
    :param vocab_size:
    :param test_split:
    :return:
    """

    probs = config.bernoulli_probability
    alpha = config.dirichlet_alpha
    sample_size = config.sample_size
    dimensions = config.dimensions
    vocab_size = config.vocab_size


    # ensure we have the probabilites
    if probs is None:
        if isinstance(alpha, float):
            alpha = torch.full((vocab_size,), alpha)
        else:
            assert len(alpha.shape) == 1
            assert alpha.size(0) == vocab_size
        # Sample from the Dirichlet distribution
        probs = torch.distributions.Dirichlet(alpha).sample([dimensions])
    else:
        assert probs.max() <= 1.
        assert probs.max() >= 0.

class CategoricalDataset(Dataset):
    def __init__(self, total_samples, probabilities):
        """
        Args:
            total_samples (int): Total number of samples to generate.
            probabilities (list): List of probabilities for each category.
        """
        super().__init__()

        self.total_samples = total_samples
        self.probabilities = probabilities
        self.num_categories = len(probabilities)
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Generate a random sample from the categorical distribution
        sample = torch.multinomial(torch.tensor(self.probabilities), 1).item()
        return sample

class StatesDataloaders2:

    def __init__(self,config):
        test_split = config.test_split
        batch_size = config.batch_size
        max_test_size = config.max_test_size

    def train():
        return None
    
    def test():
        return None

if __name__=="__main__":
    from  conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
    data_config = StatesDataloaderConfig()
    dataloader = StatesDataloader(data_config)
    databatch = next(dataloader.train().__iter__())
    x_ = databatch[0]
    print(x_)
