import os
import torch
import unittest
import numpy as np
import os
import unittest
from matplotlib import pyplot as plt
import pprint

import torch
from torch import nn

import torch.nn.functional  as F

# configs
from conditional_rate_matching.configs.config_crm import Config as ConditionalRateMatchingConfig
from conditional_rate_matching.configs.config_crm import NistConfig

#data
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders

#models
from conditional_rate_matching.models.generative_models.crm import (
    ConditionalBackwardRate,
    ClassificationForwardRate,
    conditional_transition_rate,
    telegram_bridge_probability,
    conditional_probability,
    sample_x,
    where_to_go_x,
    uniform_pair_x0_x1
)

#pipelines
from conditional_rate_matching.models.pipelines.samplers import TauLeaping

#metrics
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader
from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader

#plots
from conditional_rate_matching.utils.plots.histograms_plots import plot_marginals_binary_histograms
from conditional_rate_matching.utils.plots.histograms_plots import plot_categorical_histograms
from conditional_rate_matching.utils.plots.histograms_plots import plot_categorical_histogram_per_dimension
from conditional_rate_matching.utils.plots.paths_plots import histogram_per_dimension_plot

from conditional_rate_matching.models.metrics.crm_path_metrics import (
    telegram_bridge_probability_path,
    conditional_transition_rate_path,
    conditional_probability_path,
    classification_path,
)

if __name__=="__main__":
    config = ConditionalRateMatchingConfig(number_of_steps=10, vocab_size=4)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #conditional model
    conditional_model = ConditionalBackwardRate(config, device)
    classification_model = ClassificationForwardRate(config, device).to(device)

    # data
    dataloader_0, dataloader_1 = get_dataloaders(config)
    batch_0 , batch_1 = next(zip(dataloader_0,dataloader_1).__iter__())
    x_1, x_0 = uniform_pair_x0_x1(batch_1, batch_0,device)
    x_f, x_hist, x0_hist, ts = TauLeaping(config, classification_model, x_0, forward=True)
    x_to_go = where_to_go_x(config,x_0)

    # HISTOGRAMS
    sample_size = 100
    histogram0 = categorical_histogram_dataloader(dataloader_0, config.dimensions, config.vocab_size, maximum_test_sample_size=config.maximum_test_sample_size)
    histogram1 = categorical_histogram_dataloader(dataloader_1, config.dimensions, config.vocab_size, maximum_test_sample_size=config.maximum_test_sample_size)
    plot_categorical_histogram_per_dimension(histogram0,histogram1,torch.zeros_like(histogram0),remove_ticks=False)

    path_ = telegram_bridge_probability_path(config,ts,x_1,x_0)
    histogram_per_dimension_plot(histogram0,histogram1,path_,ts)

    #path_ = conditional_transition_rate_path()
    #path_ = conditional_probability_path()
    #path_ = classification_path()
