from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig
from conditional_rate_matching.data.graph_dataloaders_config import EgoConfig,GridConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable

from conditional_rate_matching.configs.configs_classes.config_crm import BasicTrainerConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunitySmallConfig


def experiment_ego(number_of_epochs=300,berlin=True):
    ctdd_config = CTDDConfig()
    ctdd_config.data0 = EgoConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    ctdd_config.pipeline.number_of_steps = 100
    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs, windows=berlin, metrics=[MetricsAvaliable.mse_histograms,
                                                                                                         MetricsAvaliable.graphs_plot,
                                                                                                         MetricsAvaliable.marginal_binary_histograms],
                                             learning_rate=1e-4)
    return ctdd_config

def experiment_comunity_small(number_of_epochs=300,berlin=True):
    ctdd_config = CTDDConfig()
    ctdd_config.data0 = CommunitySmallConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    ctdd_config.pipeline.number_of_steps = 100
    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs, windows=berlin, metrics=[MetricsAvaliable.mse_histograms,
                                                                                                         MetricsAvaliable.graphs_plot,
                                                                                                         MetricsAvaliable.marginal_binary_histograms],
                                             learning_rate=1e-4)
    return ctdd_config

def experiment_grid(number_of_epochs=300,berlin=True):
    ctdd_config = CTDDConfig()
    ctdd_config.data0 = GridConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    ctdd_config.pipeline.number_of_steps = 100
    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs, windows=berlin, metrics=[MetricsAvaliable.mse_histograms,
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
    config = experiment_ego(number_of_epochs=250)

    pprint(config)
    results,metrics = call_trainer(config,
                                   experiment_name="harz_experiment",
                                   experiment_type="ctdd",
                                   experiment_indentifier=None)
    print(metrics)