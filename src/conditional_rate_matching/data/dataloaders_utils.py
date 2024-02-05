from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig
from conditional_rate_matching.configs.configs_classes.config_dsb import DSBConfig
from conditional_rate_matching.data.image_dataloaders import get_data
from conditional_rate_matching.data.states_dataloaders import sample_categorical_from_dirichlet

from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.image_dataloader_config import DistortedNISTLoaderConfig
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.data.graph_dataloaders_config import BridgeConfig
from conditional_rate_matching.data.image_dataloader_config import DiscreteCIFAR10Config

from conditional_rate_matching.data.gray_codes_dataloaders import GrayCodeDataLoader
from conditional_rate_matching.data.image_dataloaders import DiscreteCIFAR10Dataloader
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders

from conditional_rate_matching.data.states_dataloaders import StatesDataloader
from conditional_rate_matching.data.ctdd_target import CTDDTargetData
from conditional_rate_matching.data.image_dataloaders import NISTLoader
from conditional_rate_matching.data.image_dataloaders import DistortedNISTLoader
from conditional_rate_matching.data.music_dataloaders import LankhPianoRollDataloader
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig

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

def get_dataloaders_crm(config:CRMConfig):
    """

    :param config:

    :return: dataloader_0,dataloader_1
    """
    #=====================================================
    # DATA 1
    #=====================================================
    if isinstance(config.data1,NISTLoaderConfig):
        dataloader_1 = NISTLoader(config.data1)
    elif isinstance(config.data1,StatesDataloaderConfig):
        dataloader_1 = StatesDataloader(config.data1)
    elif isinstance(config.data1,GraphDataloaderConfig):
        dataloader_1 = GraphDataloaders(config.data1)
    elif isinstance(config.data1,GrayCodesDataloaderConfig):
        dataloader_1 = GrayCodeDataLoader(config.data1)
    elif isinstance(config.data1,DiscreteCIFAR10Config):
        dataloader_1 = DiscreteCIFAR10Dataloader(config.data1)
    # OTHER BRIDGE ENDS
    elif isinstance(config.data1,BridgeConfig):
        dataloader_1 = GraphDataloaders(config.data0,config.data1.dataset_name)
        config.data1.dimensions = config.data0.dimensions
        config.data1.vocab_size = config.data0.vocab_size
    elif isinstance(config.data1,LakhPianoRollConfig):
        if not config.data0.conditional_model:
            piano_dataloader = LankhPianoRollDataloader(config.data1)
            dataloader_1 = piano_dataloader

    #=====================================================
    # DATA 0
    #=====================================================
    if isinstance(config.data0,NISTLoaderConfig):
        dataloader_0 = NISTLoader(config.data0)
    elif isinstance(config.data0,StatesDataloaderConfig):
        config.data0.dimensions = config.data1.dimensions
        config.data0.temporal_net_expected_shape = [config.data0.dimensions]
        config.data0.sample_size = config.data1.total_data_size
        config.data0.test_split = config.data1.test_split
        dataloader_0 = StatesDataloader(config.data0)
    elif isinstance(config.data0,GraphDataloaderConfig):
        dataloader_0 = GraphDataloaders(config.data0)
    elif isinstance(config.data0,GrayCodesDataloaderConfig):
        dataloader_0 = GrayCodeDataLoader(config.data0)
    elif isinstance(config.data0,DiscreteCIFAR10Config):
        dataloader_0 = DiscreteCIFAR10Dataloader(config.data0)
    elif isinstance(config.data0,LakhPianoRollConfig):
        if not config.data0.conditional_model:
            piano_dataloader = LankhPianoRollDataloader(config.data0)
            dataloader_0 = piano_dataloader

    # OTHER BRIDGE ENDS
    elif isinstance(config.data0,BridgeConfig):
        dataloader_0 = GraphDataloaders(config.data1,config.data0.dataset_name)
        config.data0.dimensions = config.data1.dimensions
        config.data0.vocab_size = config.data1.vocab_size

    assert config.data0.dimensions == config.data1.dimensions
    config.dimensions = config.data1.dimensions
    config.vocab_size = config.data1.vocab_size

    #============================================
    # CONDITIONAL MODEL
    #============================================

    if hasattr(config.data1,"conditional_model"):
        parent_dataloader = LankhPianoRollDataloader(config.data0)
        dataloader_0 = parent_dataloader.data0
        dataloader_1 = parent_dataloader.data1
        return dataloader_0,dataloader_1,parent_dataloader
    else:
        return dataloader_0,dataloader_1,None
    


def get_dataloaders_ctdd(config:CTDDConfig):
    """

    :param config:

    :return: dataloader_0,dataloader_1
    """
    #=====================================================
    # DATA STUFF
    #=====================================================
    if isinstance(config.data0,NISTLoaderConfig):
        dataloader_0 = NISTLoader(config.data0)
    elif isinstance(config.data0,StatesDataloaderConfig):
        dataloader_0 = StatesDataloader(config.data0)
    elif isinstance(config.data0,GraphDataloaderConfig):
        dataloader_0 = GraphDataloaders(config.data0)
    else:
        raise Exception("DataLoader not Defined")
    dataloader_1 = CTDDTargetData(config)

    return dataloader_0,dataloader_1

def get_dataloader_oops(config):
    """

    :param config:

    :return: dataloader_0,dataloader_1
    """
    # =====================================================
    # DATA STUFF
    # =====================================================
    if isinstance(config.data0, NISTLoaderConfig):
        dataloader_0 = NISTLoader(config.data0)
    elif isinstance(config.data0, StatesDataloaderConfig):
        dataloader_0 = StatesDataloader(config.data0)
    elif isinstance(config.data0, GraphDataloaderConfig):
        dataloader_0 = GraphDataloaders(config.data0)
    else:
        raise Exception("DataLoader not Defined")

    return dataloader_0

from conditional_rate_matching.data.ctdd_target_config import GaussianTargetConfig

def get_dataloaders_dsb(config:DSBConfig):
    """

    :param config:

    :return: dataloader_0,dataloader_1
    """
    #=====================================================
    # DATA STUFF
    #=====================================================
    if isinstance(config.data0,NISTLoaderConfig):
        dataloader_0 = NISTLoader(config.data0)
    elif isinstance(config.data0,StatesDataloaderConfig):
        dataloader_0 = StatesDataloader(config.data0)
    elif isinstance(config.data0,GraphDataloaderConfig):
        dataloader_0 = GraphDataloaders(config.data0)
    else:
        raise Exception("DataLoader not Defined")

    if isinstance(config.data1,NISTLoaderConfig):
        dataloader_1 = NISTLoader(config.data0)
    elif isinstance(config.data1,StatesDataloaderConfig):
        dataloader_1 = StatesDataloader(config.data0)
    elif isinstance(config.data1,GraphDataloaderConfig):
        dataloader_1 = GraphDataloaders(config.data0)
    elif isinstance(config.data1,GaussianTargetConfig):
        dataloader_1 = CTDDTargetData(config)
    else:
        raise Exception("DataLoader not Defined")


    return dataloader_0,dataloader_1