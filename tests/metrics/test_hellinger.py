import torch
import numpy as np
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig as ConditionalRateMatchingConfig

from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm

from conditional_rate_matching.models.generative_models.crm import (
    CRM
)

from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.config_files import get_experiment_dir
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
import pytest

from conditional_rate_matching.utils.plots.images_plots import plot_sample
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping
from conditional_rate_matching.configs.config_files import get_experiment_dir

def test_hellinger():
    experiment_dir = get_experiment_dir(experiment_name="prenzlauer_experiment",
                                experiment_type="crm_music",
                                experiment_indentifier="test")

    crm = CRM(experiment_dir=experiment_dir)
    data_0, data_1 = next(crm.parent_dataloader.train().__iter__())
    times = torch.rand(data_0[0].size(0))


    x_f = crm.pipeline(x_0=data_0[0])
    print(data_0[0].shape)
    print(x_f.shape)