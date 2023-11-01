import os
import torch
import unittest
import numpy as np
import os
import unittest
from matplotlib import pyplot as plt

import torch
from torch import nn

import torch.nn.functional  as F
from conditional_rate_matching.configs.config_crm import Config as ConditionalRateMatchingConfig

from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.models.generative_models.crm import conditional_probability
from conditional_rate_matching.models.generative_models.crm import telegram_bridge_probability
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from torch.utils.data import DataLoader, TensorDataset
from conditional_rate_matching.models.generative_models.crm import (
    ConditionalBackwardRate,
    ClassificationBackwardRate
)
from conditional_rate_matching.utils.plots.histograms_plots import kHistogramPlot
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader

class TestHistograms(unittest.TestCase):
    """

    """
    def test_dataloaders_histograms(self):
        config = ConditionalRateMatchingConfig(batch_size=128,sample_size=1000)

        dataloader_0,dataloader_1 = get_dataloaders(config)

        histogram = categorical_histogram_dataloader(dataloader_0, config.number_of_spins, config.number_of_states)

        kHistogramPlot(config, histogram, t=0)

        histogram = categorical_histogram_dataloader(dataloader_1, config.number_of_spins, config.number_of_states)

        kHistogramPlot(config, histogram, t=1)

if __name__=="__main__":
    print("Hello!")




