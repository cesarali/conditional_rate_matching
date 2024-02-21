import torch
from pprint import pprint
from dataclasses import dataclass,asdict
from torch.distributions import Categorical
from torch.utils.data import TensorDataset,DataLoader
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig


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
        assert probs.max() <= 1.
        assert probs.max() >= 0.

    # Sample from the categorical distribution using the Dirichlet samples as probabilities
    distribution_per_dimension = Categorical(probs)
    categorical_samples = distribution_per_dimension.sample([sample_size]).float()

    test_size = int(test_split * categorical_samples.size(0))
    train_samples = categorical_samples[test_size:]
    test_samples = categorical_samples[:test_size]

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
        return train_samples,test_samples
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

class StatesDataloader:

    name_ = "StatesDataloader"

    def __init__(self, config):
        self.config = config
        self.batch_size = config.batch_size

        self.train_loader,self.test_loader = sample_categorical_from_dirichlet(self.config)
        self.dimensions = config.dimensions

    def train(self):
        return self.train_loader

    def test(self):
        return self.test_loader

if __name__=="__main__":
    from  conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
    data_config = StatesDataloaderConfig()
    dataloader = StatesDataloader(data_config)
    databatch = next(dataloader.train().__iter__())
    x_ = databatch[0]
    print(x_)
