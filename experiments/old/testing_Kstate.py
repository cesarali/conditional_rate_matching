from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,BasicTrainerConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig


def experiment_kStates():
    """
    JUST K-States from
    :return:
    """
    crm_config = CRMConfig()
    crm_config.trainer = BasicTrainerConfig(metrics=["mse_histograms","categorical_histograms"],number_of_epochs=100)
    crm_config.data0 = StatesDataloaderConfig(dimensions=4,vocab_size=5,dirichlet_alpha=100.)
    crm_config.data1 = StatesDataloaderConfig(dimensions=4,vocab_size=5,dirichlet_alpha=0.01)
    return crm_config

