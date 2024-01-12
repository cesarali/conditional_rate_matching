import json
import torch
from typing import Union
from dataclasses import asdict

from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles


from conditional_rate_matching.models.temporal_networks.rates.ctdd_rates import GaussianTargetRateImageX0PredEMA
from conditional_rate_matching.models.temporal_networks.rates.ctdd_rates import BackRateMLP

from conditional_rate_matching.data.ctdd_target import CTDDTargetData
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_ctdd

from conditional_rate_matching.models.schedulers.scheduler import CTDDScheduler
from conditional_rate_matching.models.losses.ctdd_losses import GenericAux
from conditional_rate_matching.models.pipelines.pipeline_ctdd import CTDDPipeline
from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import GaussianTargetRate

from dataclasses import dataclass


@dataclass
class CTDD:
    """
    This class integrates all the objects required to train and generate data

    from a CTDD model, it also provides the functionality to load the models

    from the experiment files.
    """
    config: CTDDConfig = None
    experiment_dir:str = None

    experiment_files: ExperimentFiles = None
    dataloader_0: Union[GraphDataloaders] = None
    dataloader_1: CTDDTargetData = None
    backward_rate: Union[GaussianTargetRateImageX0PredEMA,BackRateMLP] = None

    process: GaussianTargetRate = None
    loss: GenericAux = None
    scheduler: CTDDScheduler = None
    pipeline: CTDDPipeline = None
    device: torch.device = None

    def __post_init__(self):
        if self.dataloader_0 is not None:
            self.pipeline = CTDDPipeline(self.config, self.backward_rate, self.dataloader_0, self.dataloader_1)
        else:
            if self.experiment_dir is not None:
                self.load_from_experiment(self.experiment_dir,self.device)
            elif self.config is not None:
                self.initialize_from_config(config=self.config,device=self.device)

    def initialize_from_config(self,config,device):
        # =====================================================
        # DATA STUFF
        # =====================================================
        self.dataloader_0, self.dataloader_1 = get_dataloaders_ctdd(config)
        # =========================================================
        # Initialize
        # =========================================================
        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.backward_rate = BackRateMLP(config, self.device).to(self.device)
        self.process = GaussianTargetRate(config, self.device)
        self.scheduler = CTDDScheduler(config,self.device)
        self.loss = GenericAux(config,device)
        self.pipeline = CTDDPipeline(self.config,self.process,self.dataloader_0,self.dataloader_1,self.scheduler)

    def load_from_experiment(self,experiment_dir,device=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        results_ = self.experiment_files.load_results()
        self.backward_rate = results_["model"]
        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = CTDDConfig(**config_path_json)

        if device is None:
            self.device = torch.device(self.config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.backward_rate.to(self.device)
        self.process = GaussianTargetRate(self.config, self.device)
        self.scheduler = CTDDScheduler(self.config,self.device)
        self.loss = GenericAux(self.config,device)
        self.dataloader_0, self.dataloader_1 = get_dataloaders_ctdd(self.config)
        self.pipeline = CTDDPipeline(self.config,self.process,self.dataloader_0,self.dataloader_1,self.scheduler)

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
