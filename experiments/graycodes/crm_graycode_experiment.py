from multiprocessing import pool
from pprint import pprint
from dataclasses import asdict
import datetime
from matplotlib.pyplot import gray
import numpy as np
import os
import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour, plot_parallel_coordinate, plot_param_importances

from conditional_rate_matching.configs.config_crm import CRMConfig, BasicTrainerConfig, ConstantProcessConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig, TemporalDeepEBMConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import AvailableGrayCodes


def CRM_single_run(dynamics="crm",
                    experiment_type="graycode",
                    experiment_indentifier="run",
                    model="mlp",
                    dataset0=None,
                    dataset1=AvailableGrayCodes.swissroll,
                    metrics=[MetricsAvaliable.marginal_binary_histograms,
                             MetricsAvaliable.grayscale_plot],
                    device="cpu",
                    epochs=100,
                    batch_size=256,
                    learning_rate=1e-3, 
                    hidden_dim=64, 
                    num_layers=2,
                    activation="ReLU",
                    time_embed_dim=8,
                    num_timesteps=64,
                    training_size=160000,
                    test_size=20000):

    databridge = dataset0 + "_" + dataset1 if dataset0 is not None else dataset1
    experiment_type = experiment_type + "_" + databridge + "_" + model + "_" + str(datetime.datetime.now().strftime("%Y.%m.%d_%Hh%Ms%S"))
    experiment_files = ExperimentFiles(experiment_name=dynamics,
                                       experiment_type=experiment_type,
                                       experiment_indentifier=experiment_indentifier,
                                       delete=True)
    #...configs:

    crm_config = CRMConfig()

    if dataset0 is None: crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., training_size=training_size, test_size=test_size, batch_size=batch_size)
    else: crm_config.data0 = GrayCodesDataloaderConfig(dataset_name=dataset0, training_size=training_size, test_size=test_size, batch_size=batch_size)
    crm_config.data1 = GrayCodesDataloaderConfig(dataset_name=dataset1, training_size=training_size, test_size=test_size, batch_size=batch_size)
    
    if model=="mlp":
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation)
        
    if model=="deepEBM":
        crm_config.temporal_network = TemporalDeepEBMConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation)
    
    crm_config.trainer = BasicTrainerConfig(number_of_epochs=epochs,
                                            learning_rate=learning_rate,
                                            device=device,
                                            metrics=metrics)
    
    crm_config.pipeline.number_of_steps = num_timesteps

    #...train

    crm = CRMTrainer(crm_config, experiment_files)
    _ , metrics = crm.train()
    return metrics


if __name__ == "__main__":


    '''
    swissroll, circles, moons, gaussians, pinwheel, spirals, checkerboard, line, cos

    '''


    CRM_single_run(dataset0=None, 
                   dataset1=AvailableGrayCodes.swissroll,
                   metrics=[MetricsAvaliable.mse_histograms,
                             MetricsAvaliable.marginal_binary_histograms,
                             MetricsAvaliable.kdmm,
                             MetricsAvaliable.grayscale_plot],
                   model="deepEBM",
                   epochs=100,
                   batch_size=128,
                   learning_rate=1e-3, 
                   hidden_dim=256, 
                   num_layers=4,
                   activation="ELU",
                   time_embed_dim=128,
                   device="cpu",
                   num_timesteps=1000)
