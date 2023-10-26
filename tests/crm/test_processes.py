import os
import unittest

import torch
from conditional_rate_matching.configs.config_crm import Config
from conditional_rate_matching.configs.config_crm import NistConfig
from conditional_rate_matching.models.generative_models.crm import sample_categorical_from_dirichlet

from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.models.generative_models.crm import conditional_probability
from conditional_rate_matching.models.generative_models.crm import telegram_bridge_probability
from conditional_rate_matching.models.pipelines.samplers import TauLeaping

from torch.utils.data import DataLoader, TensorDataset
from conditional_rate_matching.data.image_dataloaders import get_data

class TestProcesses(unittest.TestCase):
    """

    """
    def test_processes(self):
        # config = Config()
        config = NistConfig()
        config.batch_size = 64

        # =====================================================
        # DATA STUFF
        # =====================================================
        if config.dataset_name_0 == "categorical_dirichlet":
            # Parameters
            dataloader_0, _ = sample_categorical_from_dirichlet(probs=None,
                                                                alpha=config.dirichlet_alpha_0,
                                                                sample_size=config.sample_size,
                                                                dimension=config.number_of_spins,
                                                                number_of_states=config.number_of_states,
                                                                test_split=config.test_split,
                                                                batch_size=config.batch_size)

        elif config.dataset_name_0 in ["mnist", "fashion", "emnist"]:
            dataloader_0, _ = get_data(config.dataset_name_0, config)

        if config.dataset_name_1 == "categorical_dirichlet":
            # Parameters
            dataloader_1, _ = sample_categorical_from_dirichlet(probs=None,
                                                                alpha=config.dirichlet_alpha_1,
                                                                sample_size=config.sample_size,
                                                                dimension=config.number_of_spins,
                                                                number_of_states=config.number_of_states,
                                                                test_split=config.test_split,
                                                                batch_size=config.batch_size)

        elif config.dataset_name_1 in ["mnist", "fashion", "emnist"]:
            dataloader_1, _ = get_data(config.dataset_name_1, config)

        from conditional_rate_matching.utils.plots.images_plots import plot_sample
        from conditional_rate_matching.models.generative_models.crm import constant_rate
        from conditional_rate_matching.models.generative_models.crm import conditional_transition_rate

        databatch_0,databatch_1 = next(zip(dataloader_0,dataloader_1).__iter__())
        x_0 = databatch_0[0]
        x_1 = databatch_1[0]

        rate_model = lambda x, t: constant_rate(config, x, t)
        rate_model = lambda x, t: conditional_transition_rate(config, x, x_1, t)
        x_f, x_hist, x0_hist = TauLeaping(config, rate_model, x_0, forward=True)

        plot_sample(x_0)
        plot_sample(x_f)

if __name__=="__main__":
    unittest.main()