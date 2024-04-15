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

# data config
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.models.trainers.trainers_config import BasicTrainerConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.configs import temporal_network_configs


from conditional_rate_matching.models.pipelines.pipelines_config import BasicPipelineConfig

data_configs = {"NISTLoader":NISTLoaderConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "StatesDataloader":StatesDataloaderConfig,
                "GrayCodesDataloader":GrayCodesDataloaderConfig}

image_data_path = os.path.join(data_path,"raw")

@dataclass
class CTDDLossConfig:
    name :str = 'GenericAux'
    eps_ratio :float = 1e-9
    nll_weight :float = 0.001
    min_time :float = 0.01
    one_forward_pass :bool = True

@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    sampler_name:str = 'TauLeaping' # TauLeaping or PCTauLeaping

    number_of_steps:int = 20
    num_intermediates:int = 10
    sample_size:int=128
    time_epsilon = 1e-3

    min_t:float = 0.01
    eps_ratio:float = 1e-9
    initial_dist:str = 'gaussian'
    num_corrector_steps:int = 10
    corrector_step_size_multiplier:float = 1.5
    corrector_entry_time:float = 0.1

@dataclass
class CTDDConfig:

    # data
    data0: NISTLoaderConfig = NISTLoaderConfig()
    # process
    process = GaussianTargetRateConfig = GaussianTargetRateConfig()
    # temporal network
    temporal_network: Union[TemporalMLPConfig,ConvNetAutoencoderConfig] = TemporalMLPConfig()
    # loss
    loss: CTDDLossConfig = CTDDLossConfig()
    # training
    trainer: BasicTrainerConfig = BasicTrainerConfig()
    #pipeline
    pipeline : BasicPipelineConfig = BasicPipelineConfig()

    def __post_init__(self):
        if isinstance(self.data0,dict):
            self.data0 = data_configs[self.data0["name"]](**self.data0)

        if isinstance(self.temporal_network,dict):
            self.temporal_network = temporal_network_configs[self.temporal_network["name"]](**self.temporal_network)

        if isinstance(self.process,dict):
            self.process = GaussianTargetRateConfig(**self.process)

        if isinstance(self.loss,dict):
            self.loss = CTDDLossConfig(**self.loss)

        if isinstance(self.trainer,dict):
            self.trainer = BasicTrainerConfig(**self.trainer)

        if isinstance(self.pipeline,dict):
            self.pipeline = BasicPipelineConfig(**self.pipeline)