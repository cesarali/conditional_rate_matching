import os
import torch
import numpy as np
from pprint import pprint
from dataclasses import dataclass, asdict
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import torch.nn.functional as F

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,BasicPipelineConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer

from conditional_rate_matching.configs.config_files import ExperimentFiles

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMTrainerConfig

from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer

from conditional_rate_matching.models.trainers.crm_trainer import CRMDataloder
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig,LogThermostatConfig

from conditional_rate_matching.utils.plots.histogram_colors_plots import (
    plot_time_series_histograms,
    get_conditional_histograms_paths,
    categorical_counts_per_path
)

if __name__=="__main__":

    vocab_size = 6
    dimensions = 2
    num_grid_points = 100
    t_grid = torch.linspace(0.,1.,num_grid_points)

    # set dummy processes ------------------------------------------------------------
    config = CRMConfig()
    config_b = CRMConfig()
    config.data0 = StatesDataloaderConfig(dimensions=dimensions,vocab_size=vocab_size)
    config.data1 = StatesDataloaderConfig(dirichlet_alpha=0.025,dimensions=dimensions,vocab_size=vocab_size,total_data_size=1000,test_size=0.1)

    config.pipeline = BasicPipelineConfig(number_of_steps=100)
    config.thermostat = ConstantThermostatConfig(gamma=0.1)
    config.trainer = CRMTrainerConfig(number_of_epochs=100,learning_rate=1e-3)

    crm = CRM(config=config)
    crm_b = CRM(config=config_b)

    crm_b.dataloader_0 = crm.dataloader_1
    crm_b.dataloader_1 = crm.dataloader_0

    #data sample --------------------------------------------------------------------
    databatch1 = next(crm.dataloader_1.train().__iter__())
    x_1 = databatch1[0]
    databatch0 = next(crm.dataloader_0.train().__iter__())
    x_0 = databatch0[0]

    # sample forward and backward pass 
    x_f, x_path, t_path = crm.pipeline(sample_size=500,return_path=True)
    x_f, x_path_b, t_path_b = crm_b.pipeline(sample_size=500,return_path=True)

