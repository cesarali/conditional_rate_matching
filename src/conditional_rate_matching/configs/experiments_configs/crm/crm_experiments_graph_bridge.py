from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,CRMTrainerConfig
from conditional_rate_matching.data.graph_dataloaders_config import (
    EgoConfig,
    GridConfig,
    CommunitySmallConfig,
    BridgeConfig
)
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import (
    TemporalScoreNetworkAConfig,
    TemporalMLPConfig
)


"""
The following functions create config files for experiments with graph data

"""

def experiment_ego_bridge(number_of_epochs=300,berlin=True,network="mlp"):
    crm_config = CRMConfig()
    crm_config.data0 = BridgeConfig()
    crm_config.pipeline.number_of_steps = 1000
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          windows=berlin,
                                          metrics=[MetricsAvaliable.mse_histograms,
                                                   MetricsAvaliable.graphs_plot,
                                                   MetricsAvaliable.marginal_binary_histograms],
                                          learning_rate=1e-2)

    if network == "gnn":
        crm_config.data1 = EgoConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.trainer.learning_rate = 1e-2
    else:
        crm_config.data1 = EgoConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 250
        crm_config.temporal_network.time_embed_dim = 250
        crm_config.trainer.learning_rate = 1e-4

    return crm_config

def experiment_comunity_small_bridge(number_of_epochs=300,berlin=True,network="mlp"):
    crm_config = CRMConfig()
    crm_config.pipeline.number_of_steps = 1000
    crm_config.data0 = BridgeConfig()
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          windows=berlin,
                                          metrics=[MetricsAvaliable.mse_histograms,
                                                   MetricsAvaliable.graphs_plot,
                                                   MetricsAvaliable.marginal_binary_histograms],
                                          learning_rate=1e-2)

    if network == "gnn":
        crm_config.data1 = CommunitySmallConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.trainer.learning_rate = 1e-2
    else:
        crm_config.data1 = CommunitySmallConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 100
        crm_config.temporal_network.time_embed_dim = 100
        crm_config.trainer.learning_rate = 1e-4

    return crm_config

def experiment_grid_bridge(number_of_epochs=300,berlin=True,network="mlp"):
    crm_config = CRMConfig()
    crm_config.data0 = BridgeConfig()
    crm_config.pipeline.number_of_steps = 1000

    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          windows=berlin,
                                          metrics=[MetricsAvaliable.mse_histograms,
                                                   MetricsAvaliable.graphs_plot,
                                                   MetricsAvaliable.marginal_binary_histograms],
                                          learning_rate=1e-2)

    if network == "gnn":
        crm_config.data1 = GridConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.trainer.learning_rate = 1e-2
    else:
        crm_config.data1 = GridConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 100
        crm_config.temporal_network.time_embed_dim = 100
        crm_config.trainer.learning_rate = 1e-4

    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import LogThermostatConfig
    from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ExponentialThermostatConfig
    from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import \
        InvertedExponentialThermostatConfig
    from dataclasses import asdict
    from pprint import pprint

    config = experiment_comunity_small_bridge(number_of_epochs=100,network="mlp")
    #config = experiment_grid_bridge(number_of_epochs=2,network="mlp")
    #config = experiment_ego_bridge(number_of_epochs=500,network="mlp")

    #config.trainer.orca_dir = "C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/conditional_rate_matching/src/conditional_rate_matching/models/metrics/orca_berlin_2/"
    #config.trainer.windows = True

    config.thermostat = ExponentialThermostatConfig(max=1.,gamma=1.)

    config.trainer.debug = False
    config.trainer.save_model_test_stopping = True
    #config.trainer.metrics.append(MetricsAvaliable.graphs_metrics)

    config.thermostat.gamma = 2.
    config.trainer.learning_rate = 1e-3
    config.pipeline.number_of_steps = 1000
    config.trainer.loss_regularize_variance = False

    results,metrics = call_trainer(config,
                                   experiment_name="prenzlauer_experiment",
                                   experiment_type="crm",
                                   experiment_indentifier=None)
    print(metrics)
