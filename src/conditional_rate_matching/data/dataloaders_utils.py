from conditional_rate_matching.data.image_dataloaders import get_data
from conditional_rate_matching.data.states_dataloaders import sample_categorical_from_dirichlet

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
                                                           dimension=config.dimensions,
                                                           number_of_states=config.vocab_size,
                                                           test_split=config.test_split,
                                                           batch_size=config.batch_size)

    elif config.dataset_name_0 in ["mnist","fashion","emnist"]:
        dataloder_0,_ = get_data(config.dataset_name_0,config)

    if config.dataset_name_1 == "categorical_dirichlet":
        # Parameters
        dataloader_1,_ = sample_categorical_from_dirichlet(probs=None,
                                                           alpha=config.dirichlet_alpha_1,
                                                           sample_size=config.sample_size,
                                                           dimension=config.dimensions,
                                                           number_of_states=config.vocab_size,
                                                           test_split=config.test_split,
                                                           batch_size=config.batch_size)

    elif config.dataset_name_1 in ["mnist","fashion","emnist"]:
        dataloader_1,_ = get_data(config.dataset_name_1,config)

    return dataloader_0,dataloader_1