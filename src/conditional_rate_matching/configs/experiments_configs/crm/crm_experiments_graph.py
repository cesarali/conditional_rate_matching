from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,CRMTrainerConfig
from conditional_rate_matching.data.graph_dataloaders_config import (
    EgoConfig,
    GridConfig,
    CommunitySmallConfig
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

def experiment_ego(number_of_epochs=300,berlin=True,network="mlp"):
    crm_config = CRMConfig()
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)
    crm_config.pipeline.number_of_steps = 100
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          berlin=berlin,
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
        crm_config.temporal_network.hidden_dim = 50
        crm_config.temporal_network.time_embed_dim = 50
        crm_config.trainer.learning_rate = 1e-4

    return crm_config

def experiment_comunity_small(number_of_epochs=300,berlin=True,network="mlp"):
    crm_config = CRMConfig()
    crm_config.pipeline.number_of_steps = 100

    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          berlin=berlin,
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

def experiment_grid(number_of_epochs=300,berlin=True,network="mlp"):
    crm_config = CRMConfig()
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)

    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          berlin=berlin,
                                          metrics=[MetricsAvaliable.mse_histograms,
                                                   MetricsAvaliable.graphs_plot,
                                                   MetricsAvaliable.marginal_binary_histograms],
                                           learning_rate=1e-2)

    if network == "gnn":
        crm_config.data1 = GridConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.pipeline.number_of_steps = 100
        crm_config.trainer.learning_rate = 1e-2
    else:
        crm_config.data1 = GridConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 100
        crm_config.temporal_network.time_embed_dim = 100
        crm_config.trainer.learning_rate = 1e-4

    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    from dataclasses import asdict
    from pprint import pprint

    #config = experiment_comunity_small(number_of_epochs=500,network="mlp")
    #config = experiment_grid(number_of_epochs=10)
    config = experiment_ego(number_of_epochs=500,network="mlp")


    config.trainer.orca_dir = None
    config.trainer.save_model_test_stopping = True
    config.data1.init = "deg"
    config.temporal_network = TemporalMLPConfig(time_embed_dim=250,
                                                hidden_dim=250)

    config.thermostat.gamma = 1.
    config.trainer.learning_rate = 1e-3
    config.pipeline.number_of_steps = 50
    #config.trainer.metrics.append(MetricsAvaliable.loss_variance_times)
    config.trainer.loss_regularize_variance = False

    results,metrics = call_trainer(config,
                                   experiment_name="westend_experiment",
                                   experiment_type="crm",
                                   experiment_indentifier=None)
    print(metrics)