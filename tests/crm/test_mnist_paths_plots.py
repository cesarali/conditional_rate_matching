import os
import torch
import numpy as np

from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig as ConditionalRateMatchingConfig
from conditional_rate_matching.models.metrics.paths_metrics import map_proportion_nist
from conditional_rate_matching.models.metrics.paths_metrics import av_number_of_flips_in_path

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
from conditional_rate_matching.models.metrics.fid_metrics import load_classifier

def test_mnist_paths_flip_average():
    """
    av flips
    """
    ##experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5","emnist_2_mnist_lenet5","run")
    #experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_OTPlanSampler","emnist_2_mnist_lenet5_OTPlanSampler","run")
    #experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_UniformCoupling","emnist_2_mnist_lenet5_UniformCoupling","run")
    experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_OTPlanSampler_1","emnist_2_mnist_lenet5_OTPlanSampler_1","run")

    #experiment_dir = get_experiment_dir(experiment_name="prenzlauer_experiment",
    #                                    experiment_type="crm",
    #                                    experiment_indentifier="bridge_plot_mlp_mu_001")

    save_path = os.path.join(plots_path,"emnist_to_mnist_lenet5_Uniform_01.png")
    crm = CRM(experiment_dir=experiment_dir, device=torch.device("cpu"))
    av_flip = av_number_of_flips_in_path(crm,max_number_of_batches=1)
    print(av_flip)

def test_mnist_maps_batch_example():
    """
    just one sample from source and classification
    """
    from conditional_rate_matching.utils.data.samples import select_label

    number_of_images_to_see = 3
    steps_of_noise_to_see = 20

    experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5","emnist_2_mnist_lenet5","run")
    device = torch.device("cpu")
    crm = CRM(experiment_dir=experiment_dir, device=device)
    classifier = load_classifier(crm.config.data1.dataset_name,device)

    selected_images = select_label(crm.dataloader_1,label_to_see=1,sample_size=number_of_images_to_see,train=True)


    num_images_encountered = selected_images.size(0)
    crm.config.pipeline.num_intermediates = steps_of_noise_to_see
    x_f, x_hist, ts = crm.pipeline(100,return_intermediaries=True,train=False,x_0=selected_images)
    mnist_noise_bridge(x_hist, ts, steps_of_noise_to_see, min(num_images_encountered,number_of_images_to_see),
                       save_path=None)
    y = classifier(x_f.view(-1, 1, 28, 28))
    y = torch.argmax(y, dim=1)
    print(y)

def test_mnist_maps_proportions():
    ##experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5","emnist_2_mnist_lenet5","run")
    #experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_OTPlanSampler","emnist_2_mnist_lenet5_OTPlanSampler","run")
    #experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_UniformCoupling","emnist_2_mnist_lenet5_UniformCoupling","run")
    experiment_dir = os.path.join(results_path,"emnist_2_mnist_lenet5_OTPlanSampler_1","emnist_2_mnist_lenet5_OTPlanSampler_1","run")
    #experiment_dir = get_experiment_dir(experiment_name="prenzlauer_experiment",experiment_type="crm",experiment_indentifier="bridge_plot_mlp_mu_001")

    device = torch.device("cpu")
    crm = CRM(experiment_dir=experiment_dir, device=device)
    label_to_label_histograms = map_proportion_nist(crm, device, max_number_of_batches=1)
    print(label_to_label_histograms)


