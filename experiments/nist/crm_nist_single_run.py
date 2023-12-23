from multiprocessing import pool
from pprint import pprint
from dataclasses import asdict
import datetime
from matplotlib.pyplot import gray
import numpy as np
import os

from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.config_crm import CRMConfig, CRMTrainerConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig, LogThermostatConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig, ConvNetAutoencoderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig

def CRM_single_run(dynamics="crm",
                    experiment_type="nist",
                    experiment_indentifier="run",
                    thermostat=None,
                    coupling_method = 'uniform',
                    model="convnet",
                    dataset0=None,
                    dataset1="mnist",
                    metrics=[MetricsAvaliable.mse_histograms,
                             MetricsAvaliable.mnist_plot,
                             MetricsAvaliable.marginal_binary_histograms],
                    device="cpu",
                    epochs=100,
                    batch_size=64,
                    learning_rate=1e-3, 
                    hidden_dim=256,
                    time_embed_dim=128,
                    dropout=0.1,
                    num_layers=3,
                    activation="ReLU",
                    gamma=1.0,
                    num_timesteps=1000,
                   ):

    experiment_files = ExperimentFiles(experiment_name=dynamics,
                                       experiment_type=experiment_type,
                                       experiment_indentifier=experiment_indentifier,
                                       delete=True)
    #...configs:

    crm_config = CRMConfig()

    if dataset0 is None:
        crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=batch_size)

    if model=="mlp":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=True, as_image=False, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=True, as_image=False, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation,
                                                            dropout = dropout)
        
    if model=="convnet":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = ConvNetAutoencoderConfig(ema_decay = dropout,
                                                               latent_dim = hidden_dim,
                                                               decoder_channels = num_layers,
                                                               time_embed_dim = time_embed_dim,
                                                               time_scale_factor = gamma)


    if thermostat == "log":  crm_config.thermostat = LogThermostatConfig()
    else: crm_config.thermostat = ConstantThermostatConfig(gamma=gamma)

    crm_config.trainer = CRMTrainerConfig(number_of_epochs=epochs,
                                          learning_rate=learning_rate,
                                          device=device,
                                          metrics=metrics,
                                          loss_regularize_square=False,
                                          loss_regularize=False)
    
    crm_config.pipeline.number_of_steps = num_timesteps
    crm_config.optimal_transport.name = coupling_method
    #...train

    crm = CRMTrainer(crm_config, experiment_files)
    _ , metrics = crm.train()

    print('metrics=',metrics)
    return metrics


if __name__ == "__main__":

    CRM_single_run(dynamics="crm",
                   experiment_type="mnist_LogThermostat",
                   model="mlp",
                   epochs=2000,
                   thermostat="log",
                   coupling_method='uniform',
                #    dataset0="emnist",
                   dataset1="mnist",
                   metrics = ["mse_histograms", 'fid_nist', "mnist_plot", "marginal_binary_histograms"],
                   batch_size=256,
                   learning_rate=0.00155,
                   hidden_dim=85,
                   time_embed_dim=42,
                   activation="ReLU", 
                   num_layers=6,
                   dropout=0.067,
                   num_timesteps=1000,
                   device="cuda:1")
