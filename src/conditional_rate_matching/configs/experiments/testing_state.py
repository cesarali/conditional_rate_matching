from pprint import pprint
from dataclasses import asdict
from conditional_rate_matching.configs.config_crm import Config
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

from conditional_rate_matching.data.image_dataloaders import NISTLoader


def experiment_1():
    """

    :return:
    """
    crm_config = Config(metrics=["mse_histograms",
                                    "binary_paths_histograms",
                                    "marginal_binary_histograms",
                                    "mnist_plot"])
    crm_config.data0 = StatesDataloaderConfig(dimensions=784,dirichlet_alpha=100.,batch_size=128)
    crm_config.data1 = NISTLoaderConfig(batch_size=128)

    return crm_config

def experiment_2():
    """

    :return:
    """
    crm_config = Config(metrics=["mse_histograms",
                                    "binary_paths_histograms",
                                    "marginal_binary_histograms",
                                    "mnist_plot"])
    crm_config.data0 = StatesDataloaderConfig(dimensions=4,dirichlet_alpha=100.)
    crm_config.data1 = StatesDataloaderConfig(dimensions=4,dirichlet_alpha=0.01)

    return crm_config

if __name__=="__main__":
    print("Hello World!")
