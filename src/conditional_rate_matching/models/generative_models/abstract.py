import json
import torch
from torch import nn
from dataclasses import dataclass
from torch.utils.data import DataLoader

from typing import Union
from dataclasses import asdict
from torch.distributions import Categorical

from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.configs.config_crm import CRMConfig,NistConfig
from conditional_rate_matching.models.pipelines.pipeline_crm import CRMPipeline

from conditional_rate_matching.models.temporal_networks.rates.crm_rates import (
    ClassificationForwardRate,
    beta_integral
)

from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import torch
from torch.utils.data import DataLoader
import json


@dataclass
class AbstractModel(ABC):
    config: CRMConfig = None
    experiment_dir: str = None
    experiment_files: ExperimentFiles = None
    dataloader_0: Union[GraphDataloaders, DataLoader] = None
    dataloader_1: Union[GraphDataloaders, DataLoader] = None
    forward_rate: Union[ClassificationForwardRate] = None
    pipeline: CRMPipeline = None
    device: torch.device = None

    def __post_init__(self):
        super().__post_init__()
        if self.dataloader_0 is not None:
            self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1)
        else:
            if self.experiment_dir is not None:
                self.load_from_experiment(self.experiment_dir, self.device)
            elif self.config is not None:
                self.initialize_from_config(config=self.config, device=self.device)

    @abstractmethod
    def initialize_from_config(self, config, device):
        pass

    @abstractmethod
    def load_from_experiment(self, experiment_dir, device=None):
        pass

    @abstractmethod
    def start_new_experiment(self):
        pass

    def align_configs(self):
        pass