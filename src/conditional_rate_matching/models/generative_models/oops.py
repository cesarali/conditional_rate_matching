import os
import sys

import json
import torch
from torch import nn
from typing import Union
from dataclasses import asdict

from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles


from conditional_rate_matching.models.temporal_networks.rates.ctdd_rates import BackRateMLP

from conditional_rate_matching.data.ctdd_target import CTDDTargetData
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.data.dataloaders_utils import get_dataloader_oops

from conditional_rate_matching.models.losses.oops_losses import OopsEBMLoss
from conditional_rate_matching.models.pipelines.pipeline_oops import OopsPipeline
from conditional_rate_matching.models.pipelines.mc_samplers.oops_samplers import PerDimGibbsSampler,DiffSampler
from conditional_rate_matching.models.pipelines.mc_samplers.oops_samplers_utils import get_oops_samplers
from conditional_rate_matching.models.temporal_networks.ebm import EBM

from dataclasses import dataclass


@dataclass
class Oops:
    """
    This class integrates all the objects required to train and generate data

    from a CTDD model, it also provides the functionality to load the models

    from the experiment files.
    """
    config: OopsConfig = None
    experiment_dir:str = None

    experiment_files: ExperimentFiles = None
    dataloader_0: Union[GraphDataloaders] = None

    model: EBM = None
    sampler:Union[PerDimGibbsSampler,DiffSampler] = None

    loss: OopsEBMLoss = None
    pipeline: OopsPipeline = None
    device: torch.device = None

    def __post_init__(self):
        if self.experiment_dir is not None:
            self.load_from_experiment(self.experiment_dir,self.device)
        elif self.config is not None:
            self.initialize_from_config(config=self.config,device=self.device)
        else:
            raise Exception("Not Initialized")

    def initialize_from_config(self,config,device):
        # =====================================================
        # DATA STUFF
        # =====================================================
        self.dataloader_0 = get_dataloader_oops(config)
        # =========================================================
        # Initialize
        # =========================================================
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.model = EBM(self.config,device=device)
        self.sampler = get_oops_samplers(self.config)
        self.loss = OopsEBMLoss(self.config, device)
        self.pipeline = OopsPipeline(self.config,self.sampler,self.dataloader_0,self.experiment_files)
        self.pipeline.initialize(device)

    def load_from_experiment(self,experiment_dir,device=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        results_ = self.experiment_files.load_results()
        self.model = results_["model"]
        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = OopsConfig(**config_path_json)

        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.model.to(self.device)
        self.loss = OopsEBMLoss(self.config, self.device)
        self.sampler = get_oops_samplers(self.config)

        self.dataloader_0 = get_dataloader_oops(self.config)
        self.pipeline = OopsPipeline(self.config,self.sampler,self.dataloader_0,self.experiment_files)
        self.pipeline.initialize(self.device)

    def start_new_experiment(self):
        #create directories
        self.experiment_files.create_directories()
        #align configs
        self.align_configs()
        #save config
        config_as_dict = asdict(self.config)
        with open(self.experiment_files.config_path, "w") as file:
            json.dump(config_as_dict, file)

    def align_configs(self):
        pass
