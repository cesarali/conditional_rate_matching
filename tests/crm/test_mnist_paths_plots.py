import os
import torch
import numpy as np

from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig as ConditionalRateMatchingConfig

from conditional_rate_matching.models.generative_models.crm import (
    CRM
)

from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.config_files import get_experiment_dir
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.utils.plots.images_plots import plot_sample
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist

import pytest
from conditional_rate_matching.models.metrics.crm_path_metrics import conditional_bridge_marginal_probabilities_and_rates_path,conditional_bridge_images
from conditional_rate_matching.utils.plots.images_plots import mnist_grid,mnist_noise_bridge
from conditional_rate_matching import results_path
from conditional_rate_matching import plots_path

def test_mnist_paths():
    ##experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5","emnist_2_mnist_lenet5","run")
    #experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_OTPlanSampler","emnist_2_mnist_lenet5_OTPlanSampler","run")
    #experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_UniformCoupling","emnist_2_mnist_lenet5_UniformCoupling","run")
    experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_OTPlanSampler_1","emnist_2_mnist_lenet5_OTPlanSampler_1","run")

    #experiment_dir = get_experiment_dir(experiment_name="prenzlauer_experiment",
    #                                    experiment_type="crm",
    #                                    experiment_indentifier="bridge_plot_mlp_mu_001")

    save_path = os.path.join(plots_path,"emnist_to_mnist_lenet5_Uniform_01.png")
    crm = CRM(experiment_dir=experiment_dir, device=torch.device("cpu"))
    steps_of_noise_to_see = 19
    crm.config.pipeline.num_intermediates = steps_of_noise_to_see
    x_f, x_hist, ts = crm.pipeline(32,return_intermediaries=True,train=False)
    mnist_noise_bridge(x_hist, ts, steps_of_noise_to_see, 10, save_path=save_path)


