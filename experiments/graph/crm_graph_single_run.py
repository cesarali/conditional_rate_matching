from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, TemporalNetworkToRateConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig, LogThermostatConfig, ExponentialThermostatConfig,  InvertedExponentialThermostatConfig, PeriodicThermostatConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig, TemporalDeepMLPConfig
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
        crm_config.temporal_network = TemporalScoreNetworkAConfig()

    else:
        flatten = True
        full_adj = False
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation,
                                                            dropout = dropout)
        
    if dataset0 == "community_small": crm_config.data0 = CommunitySmallConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset0 == "ego": crm_config.data0 = EgoConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset0 == "grid": crm_config.data0 = GridConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    else: crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.0, batch_size=batch_size)

    if dataset1 == "community_small": crm_config.data1 = CommunitySmallConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset1 == "ego": crm_config.data1 = EgoConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset1 == "grid": crm_config.data1 = GridConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)
    elif dataset1 == "enzymes": crm_config.data1 = EnzymesConfig(flatten=flatten, as_image=False, full_adjacency=full_adj, batch_size=batch_size)

    if thermostat == "LogThermostat": crm_config.thermostat = LogThermostatConfig(time_exponential=gamma, time_base=1.0,)
    elif thermostat == "ExponentialThermostat": crm_config.thermostat = ExponentialThermostatConfig(max=max, gamma=gamma,)
    elif thermostat == "InvertedExponentialThermostat": crm_config.thermostat = InvertedExponentialThermostatConfig(max=max, gamma=gamma,)
    elif thermostat == "PeriodicThermostat": crm_config.thermostat = PeriodicThermostatConfig(max=max, gamma=gamma,)
    else: crm_config.thermostat = ConstantThermostatConfig(gamma=gamma)

    if temporal_to_rate == "linear": crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="linear",linear_reduction=.5)
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
        
        import sys

        # cuda = sys.argv[1]
        # experiment = sys.argv[2]
        # thermostat = sys.argv[3] + "Thermostat"
        # network = sys.argv[4]
        # gamma = sys.argv[5]
        # max = sys.argv[6]
        # dataset0 = experiment.split('_')[0]

        # if dataset0 == 'noise': dataset0 = None
        # coupling = 'OTPlanSampler' if experiment.split('_')[-1] == 'OT' else 'uniform'

        # CRM_single_run(dynamics="crm",
        #        experiment_type=experiment + '_' + network + '_' + thermostat + '_gamma_' + gamma + '_max_' + max,
        #        model=network,
        #        epochs=2,
        #        thermostat=thermostat+"Thermostat",
        #        coupling_method=coupling,
        #        dataset0=dataset0,
        #        dataset1="enzymes",
        #        batch_size=64,
        #        learning_rate=0.0003,
        #        num_layers=3,
        #        hidden_dim=128,
        #        time_embed_dim=128,
        #        dropout=0.1,
        #        gamma=float(gamma),
        #        max=float(max),
        #        num_timesteps=100,
        #        device="cuda:" + cuda)


        DATA = "community_small"

        CRM_single_run(dynamics="crm",
               experiment_type="graph_run" + "_" + DATA,
               model="mlp",
               thermostat="ConstantThermostat",
               coupling_method="uniform",
               dataset0=None,
               dataset1=DATA,
               metrics=[MetricsAvaliable.mse_histograms,
                        MetricsAvaliable.graphs_plot, 
                        MetricsAvaliable.graphs_metrics,
                        MetricsAvaliable.marginal_binary_histograms],
               batch_size=64,
               epochs=1000,
               learning_rate=0.0005,
               num_layers=6,
               hidden_dim=256,
               time_embed_dim=128,
               activation="Swish",
               dropout=0.1,
               gamma=1.0,
               max=0.0,
               num_timesteps=1000,
               device="cuda:2")