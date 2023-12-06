import os
from typing import List,Union
from pprint import pprint
from dataclasses import dataclass
from dataclasses import field,asdict
from conditional_rate_matching import data_path

# model config
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    ConvNetAutoencoderConfig,
)

# data config
from conditional_rate_matching.data.ctdd_target_config import GaussianTargetConfig
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig,CommunitySmallConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.models.trainers.trainers_config import BasicTrainerConfig
from conditional_rate_matching.configs import temporal_network_configs
from conditional_rate_matching.configs import thermostat_configs

from conditional_rate_matching.models.pipelines.pipelines_config import DSBPipelineConfig

from conditional_rate_matching.models.losses.dsb_losses_config import (
    RealFlipConfig,
    GradientEstimatorConfig,
    SteinSpinEstimatorConfig,
    all_flip_configs
)

from conditional_rate_matching.models.pipelines.reference_process.reference_process_config import (
    GlauberDynamicsConfig,
    GaussianTargetRateConfig,
    all_reference_process_configs
)

data_configs = {"NISTLoader":NISTLoaderConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "StatesDataloader":StatesDataloaderConfig,
                "GrayCodesDataloader":GrayCodesDataloaderConfig}

image_data_path = os.path.join(data_path,"raw")

@dataclass
class DSBTrainerConfig(BasicTrainerConfig):
    name:str = "DSBTrainer"
    number_of_sinkhorn_iterations:int = 10

@dataclass
class DSBConfig:
    # data
    data0: Union[GraphDataloaderConfig,NISTLoaderConfig] = CommunitySmallConfig(flatten=True,as_image=False,full_adjacency=False)
    data1: GaussianTargetConfig = GaussianTargetConfig()
    # temporal network
    temporal_network: Union[TemporalMLPConfig,ConvNetAutoencoderConfig] = TemporalMLPConfig()
    #flip estimator
    flip_estimator:Union[RealFlipConfig,GradientEstimatorConfig,SteinSpinEstimatorConfig] = RealFlipConfig()
    #reference
    process: Union[GaussianTargetRateConfig,GlauberDynamicsConfig] = GaussianTargetRateConfig()
    # training
    trainer: DSBTrainerConfig = DSBTrainerConfig()
    #pipeline
    pipeline = DSBPipelineConfig = DSBPipelineConfig()

    def __post_init__(self):
        if isinstance(self.data0,dict):
            self.data0 = data_configs[self.data0["name"]](**self.data0)

        if isinstance(self.data1,dict):
            self.data1 = data_configs[self.data1["name"]](**self.data1)

        if isinstance(self.temporal_network,dict):
            self.temporal_network = temporal_network_configs[self.temporal_network["name"]](**self.temporal_network)

        if isinstance(self.process, dict):
            self.process = all_reference_process_configs[self.process["name"]](**self.process)

        if isinstance(self.flip_estimator,dict):
            self.flip_estimator = all_flip_configs[self.flip_estimator["name"]](**self.flip_estimator)

        if isinstance(self.trainer,dict):
            self.trainer = DSBTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = DSBPipelineConfig(**self.pipeline)