from pprint import pprint
from dataclasses import asdict

from conditional_rate_matching.configs.config_crm import Config,BasicTrainerConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.image_dataloaders import NISTLoader


def experiment_kStates():
    """
    JUST K-States from
    :return:
    """
    crm_config = Config()
    crm_config.trainer = BasicTrainerConfig(metrics=["mse_histograms","categorical_histograms"],number_of_epochs=100)
    crm_config.data0 = StatesDataloaderConfig(dimensions=4,vocab_size=5,dirichlet_alpha=100.)
    crm_config.data1 = StatesDataloaderConfig(dimensions=4,vocab_size=5,dirichlet_alpha=0.01)
    return crm_config

