from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,BasicTrainerConfig

from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunityConfig, CommunitySmallConfig

def small_community(number_of_epochs=300,berlin=True):
    """
    :return:
    """
    crm_config = CRMConfig()

    # DATA
    crm_config.data1 = CommunitySmallConfig(batch_size=20,
                                            full_adjacency=False,
                                            flatten=True,
                                            as_image=False,
                                            max_training_size=None,
                                            max_test_size=2000)

    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)

    # RATE MODEL
    crm_config.temporal_network.hidden_dim = 50
    crm_config.temporal_network.time_embed_dim = 50

    # TRAINER
    crm_config.trainer = BasicTrainerConfig(metrics=["mse_histograms",
                                                     "binary_paths_histograms",
                                                     "marginal_binary_histograms",
                                                     "graphs_plot"],
                                            #"graphs_metrics"],
                                            number_of_epochs=number_of_epochs,
                                            learning_rate = 1e-4,
                                            windows=berlin)
    crm_config.pipeline.number_of_steps = 100
    return crm_config


def community(number_of_epochs=300,berlin=True):
    """
    :return:
    """
    crm_config = CRMConfig()

    # DATA
    crm_config.data1 = CommunityConfig(batch_size=20,
                                       full_adjacency=False,
                                       flatten=True,
                                       as_image=False,
                                       max_training_size=None,
                                       max_test_size=2000)
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)

    # RATE MODEL
    crm_config.temporal_network.hidden_dim = 50
    crm_config.temporal_network.time_embed_dim = 50

    # TRAINER
    crm_config.trainer = BasicTrainerConfig(metrics=["mse_histograms",
                                                     "binary_paths_histograms",
                                                     "marginal_binary_histograms",
                                                     "graphs_plot",
                                                     "graphs_metrics"],
                                            number_of_epochs=number_of_epochs,
                                            learning_rate = 1e-4,
                                            windows=berlin)

    crm_config.pipeline.number_of_steps = 100
    return crm_config