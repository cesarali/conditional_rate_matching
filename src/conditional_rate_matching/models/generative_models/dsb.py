import os
import re
import json
import torch

from typing import Union
from pathlib import Path
from dataclasses import asdict
from dataclasses import dataclass
from torch.utils.data import DataLoader

from conditional_rate_matching.configs.configs_classes.config_dsb import DSBConfig
from conditional_rate_matching.models.temporal_networks.rates.dsb_rate import SchrodingerBridgeRate
from conditional_rate_matching.models.pipelines.pipeline_dsb import DSBPipeline
from conditional_rate_matching.models.losses.dsb_losses import BackwardRatioSteinEstimator
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.data.transforms import SpinsToBinaryTensor
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_dsb


from conditional_rate_matching.models.pipelines.reference_process.reference_process_utils import load_reference
from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import GaussianTargetRate
from conditional_rate_matching.models.pipelines.reference_process.glauber_reference import GlauberDynamics

spins_to_binary_tensor = SpinsToBinaryTensor()

@dataclass
class DSBExperimentsFiles(ExperimentFiles):

    def set_sinkhorn(self,sinkhorn_iteration=0):
        self.best_model_path_checkpoint = os.path.join(self.experiment_dir,
                                                       "model_checkpoint_sinkhorn_{0}_{1}.tr".format(sinkhorn_iteration,
                                                                                                     "{0}"))
        self.best_model_path = os.path.join(self.experiment_dir, "best_model_sinkhorn_{0}.tr".format(sinkhorn_iteration))
        self.metrics_file = os.path.join(self.experiment_dir, "metrics_sinkhorn_{0}_{1}.json".format(sinkhorn_iteration,
                                                                                                     "{0}"))
        self.plot_path = os.path.join(self.experiment_dir, "plot_sinkhorn_{0}_{1}.png".format(sinkhorn_iteration,
                                                                                                     "{0}"))

    def extract_digits(self,s):
        pattern = Path(self.best_model_path_checkpoint)
        pattern = pattern.name
        pattern = pattern.format("(\d+)")
        match = re.match(pattern, s)
        if match is not None:
            number = int(match.group(1))
            return number
        else:
            return None


@dataclass
class DSB:
    config: DSBConfig = None
    experiment_dir:str = None

    experiment_files: DSBExperimentsFiles = None

    dataloader_0: Union[GraphDataloaders,DataLoader] = None
    dataloader_1: Union[GraphDataloaders,DataLoader] = None

    past_rate: SchrodingerBridgeRate = None
    current_rate: SchrodingerBridgeRate = None

    process: Union[GaussianTargetRate,GlauberDynamics] = None
    backward_ratio_estimator: BackwardRatioSteinEstimator = None

    pipeline: DSBPipeline = None
    device: torch.device = None
    sinkhorn_iteration: int = 0

    def __post_init__(self):
        if self.experiment_dir is not None:
            self.load_from_experiment(self.experiment_dir,self.device,sinkhorn=self.sinkhorn_iteration)
        elif self.config is not None:
            self.initialize_from_config(config=self.config,device=self.device)

    def initialize_from_config(self,config,device):
        # =====================================================
        # DATA STUFF
        # =====================================================
        self.dataloader_0, self.dataloader_1 = get_dataloaders_dsb(config)
        # =========================================================
        # Initialize
        # =========================================================
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.past_rate = SchrodingerBridgeRate(config, self.device).to(self.device)
        self.current_rate = SchrodingerBridgeRate(config, self.device).to(self.device)
        self.process = load_reference(self.config, self.device)
        self.pipeline = DSBPipeline(self.config, self.dataloader_0, self.dataloader_1,self.device)
        self.backward_ratio_estimator = BackwardRatioSteinEstimator(config, self.device)

    def load_from_experiment(self,experiment_dir,device=None,sinkhorn=0):
        self.experiment_files = DSBExperimentsFiles(experiment_dir=experiment_dir)
        self.experiment_files.set_sinkhorn(sinkhorn_iteration=sinkhorn)

        results_ = self.experiment_files.load_results()
        self.past_rate = results_["past_rate"]
        self.current_rate = results_["current_rate"]

        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = DSBConfig(**config_path_json)

        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.current_rate.to(self.device)
        self.past_rate.to(self.device)
        self.dataloader_0, self.dataloader_1 = get_dataloaders_dsb(self.config)
        self.process = load_reference(self.config,self.device)

        self.pipeline = DSBPipeline(self.config, self.dataloader_0, self.dataloader_1,self.device)
        self.backward_ratio_estimator = BackwardRatioSteinEstimator(self.config, self.device)

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