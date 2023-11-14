from conditional_rate_matching.configs.config_crm import Config
from conditional_rate_matching.data.image_dataloaders import get_data
from conditional_rate_matching.data.states_dataloaders import sample_categorical_from_dirichlet

from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig

from conditional_rate_matching.data.image_dataloaders import NISTLoader
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.data.states_dataloaders import StatesDataloader

def get_dataloaders(config):
    """

    :param config:

    :return: dataloader_0,dataloader_1
    """
    #=====================================================
    # DATA STUFF
    #=====================================================
    if config.dataset_name_0 == "categorical_dirichlet":
        # Parameters
        dataloader_0,_ = sample_categorical_from_dirichlet(probs=None,
                                                           alpha=config.dirichlet_alpha_0,
                                                           sample_size=config.sample_size,
                                                           dimensions=config.dimensions,
                                                           vocab_size=config.vocab_size,
                                                           test_split=config.test_split,
                                                           batch_size=config.batch_size)

    elif config.dataset_name_0 in ["mnist","fashion","emnist"]:
        dataloder_0,_ = get_data(config.dataset_name_0,config)

    if config.dataset_name_1 == "categorical_dirichlet":
        # Parameters
        dataloader_1,_ = sample_categorical_from_dirichlet(probs=None,
                                                           alpha=config.dirichlet_alpha_1,
                                                           sample_size=config.sample_size,
                                                           dimensions=config.dimensions,
                                                           vocab_size=config.vocab_size,
                                                           test_split=config.test_split,
                                                           batch_size=config.batch_size)

    elif config.dataset_name_1 in ["mnist","fashion","emnist"]:
        dataloader_1,_ = get_data(config.dataset_name_1,config)

    return dataloader_0,dataloader_1

def get_dataloaders_crm(config:Config):
    """

    :param config:

    :return: dataloader_0,dataloader_1
    """
    #=====================================================
    # DATA STUFF
    #=====================================================
    if isinstance(config.data1,NISTLoaderConfig):
        dataloader_1 = NISTLoader(config.data1)
    elif isinstance(config.data1,StatesDataloaderConfig):
        dataloader_1 = StatesDataloader(config.data1)
    elif isinstance(config.data1,GraphDataloaderConfig):
        dataloader_1 = GraphDataloaders(config.data1)

    if isinstance(config.data0,NISTLoaderConfig):
        dataloader_0 = NISTLoader(config.data0)
    elif isinstance(config.data0,StatesDataloaderConfig):
        config.data0.dimensions = config.data1.dimensions
        dataloader_0 = StatesDataloader(config.data0)
    elif isinstance(config.data0,GraphDataloaderConfig):
        dataloader_0 = GraphDataloaders(config.data0)

    assert config.data0.dimensions == config.data1.dimensions
    config.dimensions = config.data1.dimensions
    config.vocab_size = config.data1.vocab_size

    return dataloader_0,dataloader_1



if __name__=="__main__":
    from pprint import pprint
    from dataclasses import asdict
    from conditional_rate_matching.configs.experiments.testing_state import experiment_1
    from conditional_rate_matching.configs.experiments.testing_graphs import small_community

    config = small_community()
    config = experiment_1()

    dataloader_0,dataloader_1 = get_dataloaders_crm(config)

    pprint(config)