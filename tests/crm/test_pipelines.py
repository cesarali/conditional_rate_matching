import os
import sys
import unittest

from conditional_rate_matching.models.generative_models.crm import CRM

import torch
from torch import nn

from conditional_rate_matching.configs.config_crm import Config
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.generative_models.crm import ClassificationBackwardRate
from conditional_rate_matching.models.generative_models.crm import ConditionalBackwardRate
from conditional_rate_matching.models.pipelines.pipeline_crm import CRMPipeline

class TestCRMPipeline(unittest.TestCase):

    def test_pipeline_classifier(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        config = Config()
        dataloader_0, dataloader_1 = get_dataloaders(config)
        config.loss = "classifier"
        model = ClassificationBackwardRate(config, device).to(device)

        pipeline = CRMPipeline(config,model,dataloader_0,dataloader_1)
        x_f = pipeline(sample_size=132)
        print(x_f.shape)

    def test_pipeline_naive(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        config = Config()
        config.num_intermediates = 5

        dataloader_0, dataloader_1 = get_dataloaders(config)
        config.loss = "naive"
        model = ConditionalBackwardRate(config, device)

        pipeline = CRMPipeline(config, model, dataloader_0, dataloader_1)
        x_f = pipeline(sample_size=132)

        x_f, x_hist, ts = pipeline(sample_size=132,return_intermediaries=True)
        print(ts)


if __name__=="__main__":
    unittest.main()