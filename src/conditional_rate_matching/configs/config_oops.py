import os
from pprint import pprint
from typing import List,Union
from dataclasses import dataclass
from dataclasses import field,asdict
from conditional_rate_matching import data_path

# model config
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import(
    TemporalMLPConfig,
    ConvNetAutoencoderConfig,
)

from conditional_rate_matching.models.pipelines.reference_process.reference_process_config import GaussianTargetRateConfig
from conditional_rate_matching.models.pipelines.pipelines_config import OopsPipelineConfig

# data config
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.models.trainers.trainers_config import BasicTrainerConfig

from conditional_rate_matching.models.temporal_networks.mlp_config import ResNetEBMConfig
from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import DiffSamplerConfig,PerDimGibbsSamplerConfig


oops_mlp_configs = {
    "ResNetEBM":ResNetEBMConfig
}

oops_samplers_configs = {
    "DiffSampler":DiffSamplerConfig,
    "PerDimGibbsSampler":PerDimGibbsSamplerConfig
}


data_configs = {"NISTLoader":NISTLoaderConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "StatesDataloader":StatesDataloaderConfig}

image_data_path = os.path.join(data_path,"raw")



@dataclass
class OopsTrainer(BasicTrainerConfig):
    reinit_freq: float = 0.0
    sampling_steps: int = 100

@dataclass
class OopsLossConfig:
    name :str = 'OopsLoss'
    p_control: float = 0.0
    l2: float = 0.0

@dataclass
class OopsConfig:

    # data
    data0: NISTLoaderConfig = NISTLoaderConfig()
    # process
    sampler : Union[PerDimGibbsSamplerConfig,DiffSamplerConfig] = DiffSamplerConfig()
    # temporal network
    model_mlp: Union[ResNetEBMConfig] = ResNetEBMConfig()
    # loss
    loss: OopsLossConfig = OopsLossConfig()
    # training
    trainer: OopsTrainer = OopsTrainer()
    #pipeline
    pipeline = BasicPipelineConfig = OopsPipelineConfig()

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
            self.trainer = BasicTrainerConfig(**self.trainer)

