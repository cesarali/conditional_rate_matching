import os
from pprint import pprint
from typing import List,Union
from dataclasses import dataclass
from dataclasses import field,asdict
from conditional_rate_matching import data_path

# model config


from conditional_rate_matching.models.pipelines.pipelines_config import OopsPipelineConfig

# data config
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.models.trainers.trainers_config import BasicTrainerConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig

from conditional_rate_matching.models.temporal_networks.mlp_config import ResNetEBMConfig,MLPEBMConfig
from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import DiffSamplerConfig,PerDimGibbsSamplerConfig


oops_mlp_configs = {
    "ResNetEBM":ResNetEBMConfig,
    "MLPEBM":MLPEBMConfig
}

oops_samplers_configs = {
    "DiffSampler":DiffSamplerConfig,
    "PerDimGibbsSampler":PerDimGibbsSamplerConfig
}

data_configs = {"NISTLoader":NISTLoaderConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "StatesDataloader":StatesDataloaderConfig,
                "GrayCodesDataloader":GrayCodesDataloaderConfig}

image_data_path = os.path.join(data_path,"raw")

@dataclass
class OopsTrainerConfig(BasicTrainerConfig):
    reinit_freq: float = 0.0
    sampler_steps_per_training_iter: int = 2# 100
    test_batch_size:int = 100
    eval_every_epochs:int=10
    metrics:List = field(default_factory=lambda :[])

@dataclass
class OopsLossConfig:
    name :str = 'OopsLoss'
    p_control: float = 0.0
    l2: float = 0.0

@dataclass
class OopsConfig:

    # data
    data0: Union[NISTLoaderConfig,GraphDataloaderConfig] = NISTLoaderConfig()
    # process
    sampler : Union[PerDimGibbsSamplerConfig,DiffSamplerConfig] = PerDimGibbsSamplerConfig()
    # temporal network
    model_mlp: Union[ResNetEBMConfig,MLPEBMConfig] = ResNetEBMConfig()
    # loss
    loss: OopsLossConfig = OopsLossConfig()
    # training
    trainer: OopsTrainerConfig = OopsTrainerConfig()
    #pipeline
    pipeline = OopsPipelineConfig = OopsPipelineConfig()

    def __post_init__(self):
        if isinstance(self.data0,dict):
            self.data0 = data_configs[self.data0["name"]](**self.data0)

        if isinstance(self.model_mlp, dict):
            self.model_mlp = oops_mlp_configs[self.model_mlp["name"]](**self.model_mlp)

        if isinstance(self.sampler, dict):
            self.sampler = oops_samplers_configs[self.sampler["name"]](**self.sampler)

        if isinstance(self.pipeline,dict):
            self.pipeline = OopsPipelineConfig(**self.pipeline)

        if isinstance(self.loss,dict):
            self.loss = OopsLossConfig(**self.loss)

        if isinstance(self.trainer,dict):
            self.trainer = OopsTrainerConfig(**self.trainer)

