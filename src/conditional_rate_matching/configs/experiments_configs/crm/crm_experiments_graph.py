import os
from pprint import pprint
from dataclasses import asdict

from conditional_rate_matching.configs.config_crm import CRMConfig,CRMTrainerConfig
from conditional_rate_matching.data.graph_dataloaders_config import (
    EgoConfig,
    GridConfig,
    CommunitySmallConfig
)
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable


def experiment_ego(number_of_epochs=300,berlin=True):
    crm_config = CRMConfig()
    crm_config.data1 = EgoConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)
    crm_config.pipeline.number_of_steps = 100
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,berlin=berlin,metrics=[MetricsAvaliable.mse_histograms,
                                                                                                   MetricsAvaliable.graphs_plot,
                                                                                                   MetricsAvaliable.marginal_binary_histograms],
                                            learning_rate=1e-4)
    crm_config.temporal_network.hidden_dim = 50
    crm_config.temporal_network.time_embed_dim = 50
    return crm_config

def experiment_comunity_small(number_of_epochs=300,berlin=True):
    crm_config = CRMConfig()
    crm_config.data1 = CommunitySmallConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)
    crm_config.pipeline.number_of_steps = 100
    crm_config.temporal_network.hidden_dim = 50
    crm_config.temporal_network.time_embed_dim = 50
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,berlin=berlin,metrics=[MetricsAvaliable.mse_histograms,
                                                                                                     MetricsAvaliable.graphs_plot,
                                                                                                     MetricsAvaliable.marginal_binary_histograms],
                                            learning_rate=1e-4)
    return crm_config

def experiment_grid(number_of_epochs=300,berlin=True):
    crm_config = CRMConfig()
    crm_config.data1 = GridConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)
    crm_config.pipeline.number_of_steps = 100
    crm_config.temporal_network.hidden_dim = 50
    crm_config.temporal_network.time_embed_dim = 50
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,berlin=berlin,metrics=[MetricsAvaliable.mse_histograms,
                                                                                                   MetricsAvaliable.graphs_plot,
                                                                                                   MetricsAvaliable.marginal_binary_histograms],
                                            learning_rate=1e-4)
    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    from dataclasses import asdict
    from pprint import pprint

    config = experiment_comunity_small(number_of_epochs=500)
    #config = experiment_grid(number_of_epochs=10)
    #config = experiment_ego(number_of_epochs=500)
    #config.optimal_transport.name = "uniform"

    pprint(asdict(config))
    results,metrics = call_trainer(config,experiment_name="ot_test")
    print(metrics)