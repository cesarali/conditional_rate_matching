import json
import torch
from torch import nn
from dataclasses import dataclass
from torch.utils.data import DataLoader

from typing import Union
from dataclasses import asdict

from conditional_rate_matching.models.pipelines.pipeline_crm import CRMPipeline
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.data.music_dataloaders import LankhPianoRollDataloader
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.models.temporal_networks.rates.crm_rates import(
    ClassificationForwardRate,
)

from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.metrics.optimal_transport import OTPlanSampler
import numpy as np

@dataclass
class CRM:
    config: CRMConfig = None
    experiment_dir:str = None

    experiment_files: ExperimentFiles = None
    dataloader_0: Union[GraphDataloaders,DataLoader] = None
    dataloader_1: Union[GraphDataloaders,DataLoader] = None
    parent_dataloder: LankhPianoRollDataloader =None
    forward_rate: Union[ClassificationForwardRate] = None
    op_sampler: OTPlanSampler = None
    pipeline:CRMPipeline = None
    device: torch.device = None

    def __post_init__(self):
        self.loss = nn.CrossEntropyLoss(reduction='none')
        if self.dataloader_0 is not None:
            self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1)
        else:
            if self.experiment_dir is not None:
                self.load_from_experiment(self.experiment_dir,self.device)
            elif self.config is not None:
                self.initialize_from_config(config=self.config,device=self.device)

    def initialize_from_config(self,config,device):
        # =====================================================
        # DATA STUFF
        # =====================================================
        self.dataloader_0, self.dataloader_1,self.parent_dataloader = get_dataloaders_crm(config)
        # =========================================================
        # Initialize
        # =========================================================
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.forward_rate = ClassificationForwardRate(config, self.device).to(self.device)
        self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1)
        self.op_sampler = OTPlanSampler(**asdict(self.config.optimal_transport))

    def load_from_experiment(self,experiment_dir,device=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        results_ = self.experiment_files.load_results()

        self.forward_rate = results_["model"]

        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = CRMConfig(**config_path_json)

        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.forward_rate.to(self.device)
        self.dataloader_0, self.dataloader_1,self.parent_dataloader = get_dataloaders_crm(self.config)

        self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1,self.parent_dataloader)
        self.op_sampler = OTPlanSampler(**asdict(self.config.optimal_transport))

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

    def sample_pair(self,batch_1, batch_0,device:torch.device,seed=1980):
        x1,x0 = uniform_pair_x0_x1(batch_1, batch_0, device=torch.device("cpu"))
        x1 = x1.float()
        x0 = x0.float()
        if self.config.optimal_transport.name == "OTPlanSampler":
            batch_size = x0.size(0)
            torch.manual_seed(seed)
            np.random.seed(seed)
            #pi = self.op_sampler.get_map(x0, x1)
            #indices_i, indices_j = crm.op_sampler.sample_map(pi, batch_size=batch_size, replace=True)
            #new_x0, new_x1 = x0[indices_i], x1[indices_j]

            torch.manual_seed(seed)
            np.random.seed(seed)
            x0, x1 = self.op_sampler.sample_plan(x0, x1, replace=True)

        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        return x1,x0

def uniform_pair_x0_x1(batch_1, batch_0,device=torch.device("cpu")):
    """
    Most simple Z sampler

    :param batch_1:
    :param batch_0:

    :return:x_1, x_0
    """
    x_0 = batch_0[0]
    x_1 = batch_1[0]

    batch_size_0 = x_0.size(0)
    batch_size_1 = x_1.size(0)

    batch_size = min(batch_size_0, batch_size_1)

    x_0 = x_0[:batch_size, :]
    x_1 = x_1[:batch_size, :]

    return x_1, x_0
