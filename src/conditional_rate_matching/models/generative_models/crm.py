import json
import torch
from torch import nn
from dataclasses import dataclass
from torch.utils.data import DataLoader

from typing import Union
from dataclasses import asdict
from torch.distributions import Categorical

from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.configs.config_crm import Config,NistConfig
from conditional_rate_matching.models.pipelines.pipeline_crm import CRMPipeline

from conditional_rate_matching.models.temporal_networks.backward_rates.crm_backward_rates import (
    ClassificationForwardRate,
    beta_integral
)

from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm

@dataclass
class CRM:
    config: Config = None
    experiment_dir:str = None

    experiment_files: ExperimentFiles = None
    dataloader_0: DataLoader = None
    dataloader_1: DataLoader = None
    forward_rate: Union[ClassificationForwardRate] = None
    pipeline:CRMPipeline = None
    device: torch.device = None

    def __post_init__(self):
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
        self.dataloader_0, self.dataloader_1 = get_dataloaders_crm(config)
        # =========================================================
        # Initialize
        # =========================================================
        if device is None:
            self.device = torch.device(self.config.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.forward_rate = ClassificationForwardRate(config, self.device).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1)

    def load_from_experiment(self,experiment_dir,device=None):
        self.experiment_files = ExperimentFiles(experiment_dir=experiment_dir)
        results_ = self.experiment_files.load_results()
        self.forward_rate = results_["model"]

        config_path_json = json.load(open(self.experiment_files.config_path, "r"))
        if hasattr(config_path_json,"delete"):
            config_path_json["delete"] = False
        self.config = Config(**config_path_json)

        if device is None:
            self.device = torch.device(self.config.device) if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device

        self.forward_rate.to(self.device)

        self.dataloader_0, self.dataloader_1 = get_dataloaders_crm(self.config)

        self.loss_fn = nn.CrossEntropyLoss()
        self.pipeline = CRMPipeline(self.config, self.forward_rate, self.dataloader_0, self.dataloader_1)

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

def conditional_probability(config, x, x0, t, t0):
    """

    \begin{equation}
    P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
    \end{equation}

    \begin{equation}
    w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
    \end{equation}

    """
    right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
    right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

    t = right_time_size(t).to(x0.device)
    t0 = right_time_size(t0).to(x0.device)

    S = config.vocab_size
    integral_t0 = beta_integral(config.gamma, t, t0)

    w_t0 = torch.exp(-S * integral_t0)

    x = right_shape(x)
    x0 = right_shape(x0)

    delta_x = (x == x0).float()
    probability = 1. / S + w_t0[:, None, None] * ((-1. / S) + delta_x)

    return probability

def telegram_bridge_probability(config, x, x1, x0, t):
    """
    \begin{equation}
    P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
    \end{equation}
    """

    P_x_to_x1 = conditional_probability(config, x1, x, t=1., t0=t)
    P_x0_to_x = conditional_probability(config, x, x0, t=t, t0=0.)
    P_x0_to_x1 = conditional_probability(config, x1, x0, t=1., t0=0.)

    conditional_transition_probability = (P_x_to_x1 * P_x0_to_x) / P_x0_to_x1
    return conditional_transition_probability

def constant_rate(config, x, t):
    batch_size = x.size(0)
    dimension = x.size(1)

    rate_ = torch.full((batch_size, dimension, config.vocab_size),
                       config.gamma)
    return rate_

def where_to_go_x(config,x):
    x_to_go = torch.arange(0, config.vocab_size)
    x_to_go = x_to_go[None, None, :].repeat((x.size(0), config.dimensions, 1)).float()
    x_to_go = x_to_go.to(x.device)
    return x_to_go

def conditional_transition_rate(config, x, x1, t):
    """
    \begin{equation}
    f_t(\*x'|\*x,\*x_1) = \frac{p(\*x_1|x_t=\*x')}{p(\*x_1|x_t=\*x)}f_t(\*x'|\*x)
    \end{equation}
    """
    x_to_go = where_to_go_x(config, x)

    P_xp_to_x1 = conditional_probability(config, x1, x_to_go, t=1., t0=t)
    P_x_to_x1 = conditional_probability(config, x1, x, t=1., t0=t)

    forward_rate = constant_rate(config, x, t).to(x.device)
    rate_transition = (P_xp_to_x1 / P_x_to_x1) * forward_rate

    return rate_transition

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

    x_0 = x_0[:batch_size, :].to(device)
    x_1 = x_1[:batch_size, :].to(device)

    return x_1, x_0

def sample_x(config,x_1, x_0, time):
    device = x_1.device
    x_to_go = where_to_go_x(config, x_0)
    transition_probs = telegram_bridge_probability(config, x_to_go, x_1, x_0, time)
    sampled_x = Categorical(transition_probs).sample().to(device)
    return sampled_x

if __name__=="__main__":
    from conditional_rate_matching.configs.config_files import create_experiment_dir

    experiment_dir = create_experiment_dir(experiment_name="crm",
                                           experiment_type="dirichlet_K",
                                           experiment_indentifier="save_n_load_3")

    crm = CRM(experiment_dir=experiment_dir)