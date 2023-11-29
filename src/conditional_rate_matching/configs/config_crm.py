import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass
from dataclasses import field,asdict
from conditional_rate_matching import data_path

# model config
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    TemporalDeepMLPConfig,
    TemporalDeepSetsConfig,
    TemporalGraphConvNetConfig,
    ConvNetAutoencoderConfig,
)

# data config
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.models.trainers.trainers_config import BasicTrainerConfig
from conditional_rate_matching.configs import temporal_network_configs


data_configs = {"NISTLoader":NISTLoaderConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "StatesDataloader":StatesDataloaderConfig,
                "GrayCodesDataloader":GrayCodesDataloaderConfig}

image_data_path = os.path.join(data_path,"raw")

@dataclass
class ConstantProcessConfig:
    name:str="ConstantProcess"
    process_name:int = "constant" # constant
    gamma:float = .9

@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    number_of_steps:int = 20
    num_intermediates:int = 10

@dataclass
class CRMConfig:

    # data
    data0: StatesDataloaderConfig = StatesDataloaderConfig()
    data1: NISTLoaderConfig = NISTLoaderConfig()
    # process
    process = ConstantProcessConfig = ConstantProcessConfig()
    # temporal network
    temporal_network: Union[TemporalMLPConfig,ConvNetAutoencoderConfig] = TemporalMLPConfig()
    # training
    trainer: BasicTrainerConfig = BasicTrainerConfig()
    #pipeline
    pipeline = BasicPipelineConfig = BasicPipelineConfig()

    def __post_init__(self):
        if isinstance(self.data0,dict):
            self.data0 = data_configs[self.data0["name"]](**self.data0)

        if isinstance(self.data1,dict):
            self.data1 = data_configs[self.data1["name"]](**self.data1)

        if isinstance(self.temporal_network,dict):
            self.temporal_network = temporal_network_configs[self.temporal_network["name"]](**self.temporal_network)

        if isinstance(self.process,dict):
            self.process = ConstantProcessConfig(**self.process)

        if isinstance(self.trainer,dict):
            self.trainer = BasicTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = BasicPipelineConfig(**self.pipeline)