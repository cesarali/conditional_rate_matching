import os
import unittest

import torch
from torch import nn

from conditional_rate_matching.configs.config_crm import CRMConfig as ConditionalRateMatchingConfig

from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.models.generative_models.crm import conditional_probability
from conditional_rate_matching.models.generative_models.crm import telegram_bridge_probability
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from torch.utils.data import DataLoader, TensorDataset

from conditional_rate_matching.models.generative_models.crm import (
    CRM,
    ConditionalBackwardRate,
    ClassificationForwardRate
)

from conditional_rate_matching.configs.config_files import get_experiment_dir

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

        where_to_x = torch.arange(0, config.vocab_size)
        where_to_x = where_to_x[None, None, :].repeat((x_0.size(0), config.dimensions, 1)).float()
        where_to_x = where_to_x.to(x_0.device)

        probs = conditional_probability(config, where_to_x, x_0, time, t0=0.)
        probs_transition = telegram_bridge_probability(config, where_to_x, x_1, x_0, time)

        print(probs.sum(axis=-1))
        print(probs_transition.sum(axis=-1))

    def test_model_rates_classifier(self):
        config = ConditionalRateMatchingConfig()
        dataloader_0,dataloader_1 = get_dataloaders(config)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        config.loss = "classifier"

        model = ClassificationForwardRate(config, device).to(device)

        batch_1, batch_0 = next(zip(dataloader_1, dataloader_0).__iter__())
        x_0 = batch_0[0].to(device)
        x_1 = batch_1[0].to(device)
        time = torch.rand((x_0.size(0))).to(device)

        rates_ = model(x_0,time)
        is_positive = torch.all(rates_.gt(0))
        print(rates_.shape)
        print(is_positive)

    @unittest.skip
    def test_model_rates(self):
        config = ConditionalRateMatchingConfig()
        dataloader_0,dataloader_1 = get_dataloaders(config)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        config.loss = "naive"
        model = ConditionalBackwardRate(config, device)
        loss_fn = nn.MSELoss()

        batch_1, batch_0 = next(zip(dataloader_1, dataloader_0).__iter__())
        x_0 = batch_0[0].to(device)
        x_1 = batch_1[0].to(device)
        time = torch.rand((x_0.size(0))).to(device)

        rates_ = model(x_0,time)
        is_positive = torch.all(rates_.gt(0))
        print(rates_.shape)
        print(is_positive)

class TestCRMLoading(unittest.TestCase):

    def test_load(self):
        from conditional_rate_matching.models.metrics.crm_metrics_utils import log_metrics
        from conditional_rate_matching.utils.plots.images_plots import mnist_grid

        experiment_dir = get_experiment_dir(experiment_name="crm",
                                            experiment_type="mnist",
                                            experiment_indentifier="save_n_loads8")

        crm = CRM(experiment_dir=experiment_dir,device=torch.device("cpu"))
        generative_sample = crm.pipeline(32)
        mnist_grid(generative_sample)

        """
        log_metrics(crm,
                    epoch=None,
                    metrics_to_log=["binary_paths_histograms"],
                    where_to_log={"binary_paths_histograms":None})
        """


if __name__=="__main__":
    unittest.main()