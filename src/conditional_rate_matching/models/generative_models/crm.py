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
from torch.nn.functional import softmax

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
    image_data_path: str = None

    def __post_init__(self):
        self.loss = nn.CrossEntropyLoss(reduction='none')
        if self.dataloader_0 is not None:
            self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1)
        else:
            if self.experiment_dir is not None:
                self.load_from_experiment(self.experiment_dir,self.device,self.image_data_path)
            elif self.config is not None:
                self.initialize_from_config(config=self.config,device=self.device)

    def initialize_from_config(self,config,device):
        # =====================================================
        # DATA STUFF
        # =====================================================
        self.dataloader_0,self.dataloader_1,self.parent_dataloader = get_dataloaders_crm(config)
        # =========================================================
        # Initialize
        # =========================================================
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.forward_rate = ClassificationForwardRate(self.config, self.device).to(self.device)
        self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1,self.parent_dataloader)
        if self.config.optimal_transport.cost == "log":
            B = self.forward_rate.log_cost_regularizer()
            B = B.item() if isinstance(B,torch.Tensor) else B
            self.config.optimal_transport.method = "sinkhorn"
            self.config.optimal_transport.normalize_cost = True
            self.config.optimal_transport.normalize_cost_constant = float(self.config.data1.dimensions)
            reg = 1./B
            print("OT regularizer for Schrodinger Plan {0}".format(reg))
            self.config.optimal_transport.reg = reg

        self.op_sampler = OTPlanSampler(**asdict(self.config.optimal_transport))

    def load_from_experiment(self,experiment_dir,device=None,set_data_path=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        results_ = self.experiment_files.load_results()

        
        self.forward_rate = results_["model"]

        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = CRMConfig(**config_path_json)
        if set_data_path is not None:
            self.config.data1.data_dir = set_data_path
            self.config.data0.data_dir = set_data_path
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

    def sample_pair(self,batch_1, batch_0,device:torch.device,seed=None):
        x1,x0 = uniform_pair_x0_x1(batch_1, batch_0, device=torch.device("cpu"))
        x1 = x1.float()
        x0 = x0.float()

        batch_size = x0.shape[0]
        x0 = x0.reshape(batch_size,-1)
        x1 = x1.reshape(batch_size,-1)
        
        if self.config.optimal_transport.name == "OTPlanSampler":

            cost=None
            if self.config.optimal_transport.cost == "log":
                with torch.no_grad():
                    cost = self.forward_rate.log_cost(x0,x1)

            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                
            x0, x1 = self.op_sampler.sample_plan(x0, x1, replace=False,cost=cost)

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
