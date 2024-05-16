from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, TemporalNetworkToRateConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig, LogThermostatConfig, ExponentialThermostatConfig,  InvertedExponentialThermostatConfig, PeriodicThermostatConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig, TemporalDeepMLPConfig, SimpleTemporalGCNConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunitySmallConfig, EgoConfig, GridConfig, EnzymesConfig


""" Default configurations for training.
"""

def CRM_single_run(dynamics="crm",
                    experiment_type="graph",
                    experiment_indentifier="run",
                    thermostat=None,
                    coupling_method='uniform', # uniform, OTPlanSampler
                    model="mlp",
                    dataset0=None,
                    dataset1="ego",
                    metrics=[MetricsAvaliable.mse_histograms,
                             MetricsAvaliable.graphs_plot, 
                             MetricsAvaliable.graphs_metrics,
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
                    gamma =1.0,
                    max=1.0,
                    num_timesteps=50,
                    temporal_to_rate=None
                    ):

    experiment_files = ExperimentFiles(experiment_name=dynamics,
                                       experiment_type=experiment_type,
                                       experiment_indentifier=experiment_indentifier,
                                       delete=True)
    #...configs:

    crm_config = CRMConfig()

    # if dataset0 is None:
    #     crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=batch_size)

    if model=="gnn":
        flatten = False
        full_adj = True
        crm_config.temporal_network = SimpleTemporalGCNConfig(hidden_channels=hidden_dim,
                                                              time_embed_dim = time_embed_dim,)


    else:
        flatten = True
        full_adj = False
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation,
                                                            dropout = dropout)
        
    if dataset0 == "community": crm_config.data0 = CommunitySmallConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset0 == "ego": crm_config.data0 = EgoConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset0 == "grid": crm_config.data0 = GridConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    else: crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.0, batch_size=batch_size)

    if dataset1 == "community": crm_config.data1 = CommunitySmallConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset1 == "ego": crm_config.data1 = EgoConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset1 == "grid": crm_config.data1 = GridConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset1 == "enzymes": crm_config.data1 = EnzymesConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)

    if thermostat == "LogThermostat": crm_config.thermostat = LogThermostatConfig(time_exponential=gamma, time_base=1.0,)
    elif thermostat == "ExponentialThermostat": crm_config.thermostat = ExponentialThermostatConfig(max=max, gamma=gamma,)
    elif thermostat == "InvertedExponentialThermostat": crm_config.thermostat = InvertedExponentialThermostatConfig(max=max, gamma=gamma,)
    elif thermostat == "PeriodicThermostat": crm_config.thermostat = PeriodicThermostatConfig(max=max, gamma=gamma,)
    else: crm_config.thermostat = ConstantThermostatConfig(gamma=gamma)

    if temporal_to_rate == "linear": crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="linear",linear_reduction=.25)
    elif temporal_to_rate == "bernoulli": crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="bernoulli")
    else: crm_config.temporal_network_to_rate = None

    crm_config.trainer = CRMTrainerConfig(number_of_epochs=epochs,
                                          learning_rate=learning_rate,
                                          device=device,
                                          metrics=metrics,
                                          loss_regularize_square=False,
                                          loss_regularize=False)
    crm_config.trainer.orca_dir = "/home/df630/conditional_rate_matching/src/conditional_rate_matching/models/metrics/orca_new_jersey"
    crm_config.trainer.windows = False
    crm_config.pipeline.number_of_steps = num_timesteps
    crm_config.optimal_transport.name = coupling_method

    #...train

    crm = CRMTrainer(crm_config, experiment_files)
    _ , metrics = crm.train()

    print('metrics=',metrics)
    return metrics


if __name__ == "__main__":
        
    import argparse
    import datetime
    import os
    import torch

    date = datetime.datetime.now().strftime("%Hh%Ms%S_%Y.%m.%d")

    if torch.cuda.is_available():

        #..Parse the arguments

        parser = argparse.ArgumentParser(description='Run the CRM training with specified configurations.')
        parser.add_argument('--id', type=str, required=False, help='Experiment indentifier', default='run')
        parser.add_argument('--source', type=str, required=True, help='Source dataset')
        parser.add_argument('--target', type=str, required=True, help='Target dataset')
        parser.add_argument('--model', type=str, required=True, help='Model for the network')
        parser.add_argument('--timesteps', type=int, required=False, help='Number of timesteps', default=1000)
        parser.add_argument('--timepsilon', type=float, required=False, help='Stop at time t=1-epsilon from target', default=None)        
        parser.add_argument('--coupling', type=str, required=False, help='Type of source-target coupling', default='uniform')
        parser.add_argument('--epochs', type=int, required=False, help='Number of epochs', default=1000)
        parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=16)
        parser.add_argument('--dim', type=int, required=False, help='Hidden dimension size', default=128)
        parser.add_argument('--temb', type=int, required=False, help='Dimension of time emebedding', default=32)
        parser.add_argument('--act', type=str, required=False, help='Activation function', default='Swish')
        parser.add_argument('--lr', type=float, required=False, help='Learning rate', default=0.001)
        parser.add_argument('--dropout', type=float, required=False, help='Dropout rate', default=0.1)
        parser.add_argument('--thermostat', type=str, required=False, help='Type of thermostat', default='Constant')
        parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for thermostat', default=None)
        parser.add_argument('--max', type=float, required=False, help='Max parameter for thermostat', default=None)
        parser.add_argument('--slope', type=float, required=False, help='Slope parameter for thermostat', default=None)
        parser.add_argument('--shift', type=float, required=False, help='Shift parameter for thermostat', default=None)
        parser.add_argument('--exponent', type=float, required=False, help='Exponent parameter for thermostat', default=None)
        parser.add_argument('--logexp', type=float, required=False, help='Exponential parameter for thermostat', default=None)
        parser.add_argument('--timebase', type=float, required=False, help='Time base parameter for thermostat', default=None)
        parser.add_argument('--device', type=str, required=False, help='Selected device', default="cuda:0")

        arg = parser.parse_args()

        params = ['gamma', 'max', 'slope', 'shift', 'exponent', 'logexp', 'timebase']
        therm_params = {}
        for p in params:
            if p in vars(arg).keys():
                if vars(arg)[p] is not None:
                    therm_params[p] = vars(arg)[p]

        full_experiment_type = f"graphs_{arg.source}_to_{arg.target}_{arg.model}_{arg.coupling}_coupling_{arg.thermostat}Thermostat" + "_" + "_".join([f"{k}_{v}" for k,v in therm_params.items()]) + "__" + date 

        CRM_single_run(dynamics="crm",
                        experiment_type=full_experiment_type,
                        experiment_indentifier=arg.id,
                        model=arg.model,
                        thermostat=arg.thermostat + "Thermostat",
                        coupling_method="uniform",
                        dataset0=arg.source,
                        dataset1=arg.target,
                        metrics=[MetricsAvaliable.mse_histograms,
                                 MetricsAvaliable.graphs_plot, 
                                 MetricsAvaliable.graphs_metrics,
                                 MetricsAvaliable.marginal_binary_histograms],
                        batch_size=arg.batch_size,
                        epochs=arg.epochs,
                        learning_rate=arg.lr,
                        hidden_dim=arg.dim,
                        time_embed_dim=arg.temb,
                        dropout=arg.dropout,
                        gamma=arg.gamma,
                        max=arg.max,
                        num_timesteps=arg.timesteps,
                        device=arg.device)