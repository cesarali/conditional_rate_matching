import os
import unittest

import torch
from conditional_rate_matching.configs.config_crm import Config as ConditionalRateMatchingConfig

from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.models.generative_models.crm import conditional_probability
from conditional_rate_matching.models.generative_models.crm import telegram_bridge_probability
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from torch.utils.data import DataLoader, TensorDataset


class TestCRM(unittest.TestCase):
    """

    """
    def test_conditional_probability(self):
        config = ConditionalRateMatchingConfig()
        dataloader_0,dataloader_1 = get_dataloaders(config)

        batch_1, batch_0 = next(zip(dataloader_1, dataloader_0).__iter__())
        x_0 = batch_0[0]
        x_1 = batch_1[0]
        time = torch.rand((x_0.size(0)))
        x_1,x_0 = uniform_pair_x0_x1(batch_1,batch_0)

        where_to_x = torch.arange(0, config.number_of_states)
        where_to_x = where_to_x[None, None, :].repeat((x_0.size(0), config.number_of_spins, 1)).float()
        where_to_x = where_to_x.to(x_0.device)

        probs = conditional_probability(config, where_to_x, x_0, time, t0=0.)
        probs_transition = telegram_bridge_probability(config, where_to_x, x_1, x_0, time)

        print(probs.sum(axis=-1))
        print(probs_transition.sum(axis=-1))

if __name__=="__main__":
    unittest.main()