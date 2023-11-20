import torch
import numpy as np

from typing import Union
from torchtyping import TensorType
from conditional_rate_matching.configs.config_ctdd import CTDDConfig
from conditional_rate_matching.data.transforms import BinaryTensorToSpinsTransform


class CTDDTargetData():
    config : Union[CTDDConfig]

    doucet:bool = True

    def __init__(self,config:CTDDConfig,rank=None):
        self.config = config

        self.dimensions = self.config.data0.dimensions
        self.shape = self.config.data0.temporal_net_expected_shape

        self.S = self.config.data0.vocab_size
        sampler_config = self.config.pipeline

        self.initial_dist = sampler_config.initial_dist
        if self.initial_dist == 'gaussian':
            self.initial_dist_std = self.config.process.Q_sigma
        else:
            self.initial_dist_std = None

    def sample(self, num_of_paths:int, device=None) -> TensorType["num_of_paths","D"]:

        if self.initial_dist == 'uniform':
            x = torch.randint(low=0, high=self.S, size=(num_of_paths, self.dimensions), device=device)
        elif self.initial_dist == 'gaussian':
            target = np.exp(
                - ((np.arange(1, self.S + 1) - self.S // 2) ** 2) / (2 * self.initial_dist_std ** 2)
            )
            target = target / np.sum(target)

            cat = torch.distributions.categorical.Categorical(
                torch.from_numpy(target)
            )
            x = cat.sample((num_of_paths * self.dimensions,)).view(num_of_paths, self.dimensions)
        else:
            raise NotImplementedError('Unrecognized initial dist ' + self.initial_dist)

        return [x,None]

    def train(self):

        training_size = self.config.data0.training_size
        batch_size = self.config.data0.batch_size

        current_index = 0
        while current_index < training_size:
            remaining = min(training_size - current_index, batch_size)
            x = self.sample(remaining)
            # Your processing code here
            current_index += remaining
            yield x

    def test(self):
        test_size = self.config.data0.test_size
        batch_size = self.config.data0.batch_size

        number_of_batches = int(test_size / batch_size) + 1
        for a in range(number_of_batches):
            x = self.sample(batch_size)
            yield x
