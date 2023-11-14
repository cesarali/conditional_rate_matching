from pprint import pprint
from dataclasses import asdict
from conditional_rate_matching.configs.config_crm import Config
from conditional_rate_matching.data.image_dataloaders import NISTLoader

from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunityConfig, CommunitySmallConfig

def small_community():
    """

    :return:
    """
    crm_config = Config(metrics=["mse_histograms",
                                 "binary_paths_histograms",
                                 "marginal_binary_histograms"])
    crm_config.data1 = CommunitySmallConfig(batch_size=128,
                                            full_adjacency=False,
                                            flatten=True,as_image=False,
                                            max_training_size=None,
                                            max_test_size=2000)
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=128)
    crm_config.temporal_network.hidden_dim = 500
    crm_config.temporal_network.time_embed_dim = 250
    crm_config.learning_rate = 1e-4


    return crm_config

if __name__=="__main__":
    import os
    import yaml
    from dataclasses import asdict
    from conditional_rate_matching import config_path

    config = small_community()
    data = asdict(config)

    with open(os.path.join(config_path,"crm","small_community.yml"), 'w') as file:
        yaml.dump(data, file)
