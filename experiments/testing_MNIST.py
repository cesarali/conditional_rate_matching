from pprint import pprint
from dataclasses import asdict
from conditional_rate_matching.configs.config_crm import Config,BasicTrainerConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.image_dataloaders import NISTLoader


from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    ConvNetAutoencoderConfig,
    TemporalMLPConfig
)

def experiment_MNIST(max_training_size=60000,max_test_size=5000,berlin=True):
    """
    MNIST EXPERIMENT
    :return:
    """
    crm_config = Config()
    crm_config.trainer = BasicTrainerConfig(metrics=["mse_histograms","binary_paths_histograms",
                                                     "marginal_binary_histograms","mnist_plot"],
                                            number_of_epochs=10,berlin=berlin)
    
    crm_config.data1 = NISTLoaderConfig(batch_size=128,
                                        max_training_size=max_training_size,
                                        max_test_size=max_test_size)
    crm_config.data0 = StatesDataloaderConfig(dimensions=784,
                                              dirichlet_alpha=100.,
                                              batch_size=128,
                                              sample_size=max_training_size,
                                              max_test_size=max_test_size)

    crm_config.temporal_network.hidden_dim = 500
    crm_config.temporal_network.time_embed_dim = 250

    return crm_config


def experiment_MNIST_Convnet(max_training_size=60000,max_test_size=5000,berlin=True):
    """
    MNIST EXPERIMENT
    :return:
    """
    crm_config = Config()
    crm_config.trainer = BasicTrainerConfig(metrics=["mse_histograms","binary_paths_histograms",
                                                     "marginal_binary_histograms","mnist_plot"],
                                            number_of_epochs=4,
                                            berlin=berlin)

    crm_config.data1 = NISTLoaderConfig(batch_size=128,
                                        max_training_size=max_training_size,
                                        as_image=True,
                                        flatten=False,
                                        max_test_size=max_test_size)

    crm_config.data0 = StatesDataloaderConfig(dimensions=784,
                                              dirichlet_alpha=100.,
                                              batch_size=128,
                                              sample_size=max_training_size*10,
                                              max_test_size=max_test_size)

    crm_config.temporal_network = ConvNetAutoencoderConfig()


    return crm_config