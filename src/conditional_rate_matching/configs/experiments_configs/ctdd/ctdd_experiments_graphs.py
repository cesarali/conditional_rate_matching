import os
from pprint import pprint
from dataclasses import asdict

from conditional_rate_matching.configs.config_ctdd import CTDDConfig
from conditional_rate_matching.data.graph_dataloaders_config import EgoConfig,GridConfig,CommunitySmallConfig
from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import DiffSamplerConfig,PerDimGibbsSamplerConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable

from pprint import pprint
from dataclasses import asdict
from conditional_rate_matching.data.image_dataloaders import NISTLoader
from conditional_rate_matching.configs.config_crm import CRMConfig,BasicTrainerConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunitySmallConfig


def experiment_ego(number_of_epochs=300,berlin=True):
    ctdd_config = CTDDConfig()
    ctdd_config.data0 = EgoConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    ctdd_config.pipeline.number_of_steps = 100
    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs,berlin=berlin,metrics=[MetricsAvaliable.mse_histograms,
                                                                                                     MetricsAvaliable.graphs_plot,
                                                                                                     MetricsAvaliable.marginal_binary_histograms],
                                             learning_rate=1e-4)
    return ctdd_config

def experiment_comunity_small(number_of_epochs=300,berlin=True):
    ctdd_config = CTDDConfig()
    ctdd_config.data0 = CommunitySmallConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    ctdd_config.pipeline.number_of_steps = 100
    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs,berlin=berlin,metrics=[MetricsAvaliable.mse_histograms,
                                                                                                     MetricsAvaliable.graphs_plot,
                                                                                                     MetricsAvaliable.marginal_binary_histograms],
                                             learning_rate=1e-4)
    return ctdd_config

def experiment_grid(number_of_epochs=300,berlin=True):
    ctdd_config = CTDDConfig()
    ctdd_config.data0 = GridConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    ctdd_config.pipeline.number_of_steps = 100
    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs,berlin=berlin,metrics=[MetricsAvaliable.mse_histograms,
                                                                                                     MetricsAvaliable.graphs_plot,
                                                                                                     MetricsAvaliable.marginal_binary_histograms],
                                             learning_rate=1e-4)
    return ctdd_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    from dataclasses import asdict
    from pprint import pprint

    #config = experiment_comunity_small(number_of_epochs=500)
    #config = experiment_grid(number_of_epochs=10)
    config = experiment_ego(number_of_epochs=10)

    pprint(asdict(config))
    results,metrics = call_trainer(config)
    print(metrics)