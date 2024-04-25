import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass
from dataclasses import field,asdict
from conditional_rate_matching import data_path

from conditional_rate_matching.models.networks.mlp_config import MLPConfig

# model config
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    TemporalDeepMLPConfig,
    TemporalLeNet5Config,
    TemporalLeNet5AutoencoderConfig,
    TemporalUNetConfig,
    # TemporalDeepSetsConfig,
    TemporalGraphConvNetConfig,
    ConvNetAutoencoderConfig,
    DiffusersUnet2DConfig,
    TemporalScoreNetworkAConfig,
    SequenceTransformerConfig
)

# data config
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.image_dataloaders_conditional_config import DistortedNISTLoaderConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.models.trainers.trainers_config import BasicTrainerConfig
from conditional_rate_matching.configs import temporal_network_configs,conditional_network_configs
from conditional_rate_matching.configs import thermostat_configs
from conditional_rate_matching.models.pipelines.pipelines_config import BasicPipelineConfig
from conditional_rate_matching.data.graph_dataloaders_config import BridgeConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig,LogThermostatConfig

data_configs = {"NISTLoader":NISTLoaderConfig,
                "DistortedNISTLoader":DistortedNISTLoaderConfig,
                "LakhPianoRoll":LakhPianoRollConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "BridgeConfig":BridgeConfig,
                "StatesDataloader":StatesDataloaderConfig,
                "GrayCodesDataloader":GrayCodesDataloaderConfig}

image_data_path = os.path.join(data_path,"raw")

@dataclass
class CRMTrainerConfig(BasicTrainerConfig):
    name:str = "CRMTrainer"
    loss_regularize_variance:bool = False
    loss_regularize:bool = False
    loss_regularize_square:bool = False

@dataclass
class OptimalTransportSamplerConfig:
    name: str = "uniform" # uniform,OTPlanSampler
    method: str = "exact"
    reg: float = 0.05
    reg_m: float = 1.0
    normalize_cost: bool = False
    warn: bool = True

@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    number_of_steps:int = 20
    num_intermediates:int = 10
    time_epsilon = 0.05

@dataclass
class TemporalNetworkToRateConfig:
    name:str = "TemporalNetworkToRate"
    type_of:str = None # bernoulli, empty, linear, None
    linear_reduction:Union[float,int] = 0.1 # if None full layer, 
                                            # if float is the percentage of output dimensions that is assigned as hidden 
                                            #otherwise hidden

@dataclass
class CRMConfig:
    # data
    data0: Union[LakhPianoRollConfig,StatesDataloaderConfig] = StatesDataloaderConfig()
    data1: Union[LakhPianoRollConfig,NISTLoaderConfig,StatesDataloaderConfig] = NISTLoaderConfig()
    # process
    thermostat : Union[ConstantThermostatConfig, LogThermostatConfig] = ConstantThermostatConfig()
    # temporal_to_rate
    temporal_network_to_rate : Union[int,float,TemporalNetworkToRateConfig] = None
    # temporal network
    temporal_network: Union[TemporalMLPConfig,ConvNetAutoencoderConfig,DiffusersUnet2DConfig,TemporalScoreNetworkAConfig,SequenceTransformerConfig] = TemporalMLPConfig()
    # conditional model
    conditional_network: MLPConfig = None
    # ot
    optimal_transport:OptimalTransportSamplerConfig = OptimalTransportSamplerConfig()
    # training
    trainer: CRMTrainerConfig = CRMTrainerConfig()
    #pipeline
    pipeline : BasicPipelineConfig = BasicPipelineConfig()

    def __post_init__(self):
        if isinstance(self.data0,dict):
            self.data0 = data_configs[self.data0["name"]](**self.data0)

        if isinstance(self.data1,dict):
            self.data1 = data_configs[self.data1["name"]](**self.data1)

        if isinstance(self.temporal_network,dict):
            self.temporal_network = temporal_network_configs[self.temporal_network["name"]](**self.temporal_network)

        if isinstance(self.conditional_network,dict):
            self.temporal_network = conditional_network_configs[self.conditional_network["name"]](**self.conditional_network)

        if isinstance(self.optimal_transport,dict):
            self.optimal_transport = OptimalTransportSamplerConfig(**self.optimal_transport)

        if isinstance(self.thermostat, dict):
            self.thermostat = thermostat_configs[self.thermostat["name"]](**self.thermostat)

        if isinstance(self.trainer,dict):
            self.trainer = CRMTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = BasicPipelineConfig(**self.pipeline)

        if isinstance(self.temporal_network_to_rate,dict):
            self.temporal_network_to_rate = TemporalNetworkToRateConfig(**self.temporal_network_to_rate)
