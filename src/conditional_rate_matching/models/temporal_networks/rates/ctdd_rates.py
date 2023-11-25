import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


from conditional_rate_matching.configs.config_ctdd import CTDDConfig

from typing import Union, Tuple
from torchtyping import TensorType
from conditional_rate_matching.models.temporal_networks.embedding_utils import transformer_timestep_embedding
from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import GaussianTargetRate

from dataclasses import dataclass
from diffusers.utils import BaseOutput
from conditional_rate_matching.models.temporal_networks.ema import EMA
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network
from abc import ABC,abstractmethod
from functools import reduce

@dataclass
class BackwardRateOutput(BaseOutput):
    """
    :param BaseOutput:
    :return:
    """
    x_logits: torch.Tensor
    p0t_reg: torch.Tensor
    p0t_sig: torch.Tensor
    reg_x: torch.Tensor

class BackwardRate(nn.Module,ABC):

    def __init__(self,
                 config:CTDDConfig,
                 device,
                 rank,
                 **kwargs):
        super().__init__()

        self.config = config

        # DATA
        self.dimensions = config.data0.dimensions
        self.vocab_size = config.data0.vocab_size
        self.data_min_max = config.data0.data_min_max

        # TIME
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.act = nn.functional.silu

    def init_parameters(self):
        return None

    @abstractmethod
    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"]
                )-> TensorType["batch_size", "dimension", "num_states"]:
        return None

    def _center_data(self, x):
        out = (x - self.data_min_max[0]) / (self.data_min_max[1] - self.data_min_max[0])  # [0, 1]
        return 2 * out - 1  # to put it in [-1, 1]

    def forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"],
                x_tilde: TensorType["batch_size", "dimension"] = None,
                return_dict: bool = False,
                )-> Union[BackwardRateOutput, torch.FloatTensor, Tuple]:
        if x_tilde is not None:
            return self.ctdd(x,x_tilde,times,return_dict)
        else:
            x_logits = self._forward(x, times)
            if not return_dict:
                return x_logits
            else:
                return BackwardRateOutput(x_logits=x_logits, p0t_reg=None, p0t_sig=None)

    def ctdd(self,x_t,x_tilde,times,return_dict):
        if x_tilde is not None:
            if self.config.loss.one_forward_pass:
                reg_x = x_tilde
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = p0t_reg
            else:
                reg_x = x_t
                x_logits = self._forward(reg_x, times)  # (B, D, S)
                p0t_reg = F.softmax(x_logits, dim=2)  # (B, D, S)
                p0t_sig = F.softmax(self._forward(x_tilde, times), dim=2)  # (B, D, S)

            if not return_dict:
                return (x_logits,p0t_reg,p0t_sig,reg_x)
            return BackwardRateOutput(x_logits=x_logits,p0t_reg=p0t_reg,p0t_sig=p0t_sig,reg_x=reg_x)
        else:
            return self._forward(x_t, times)  # (B, D, S)

    def flip_rate(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension"]:
        forward_logits = self.forward(x,times)
        return forward_logits[:,:,0]

class ImageX0PredBase(BackwardRate):
    def __init__(self, cfg:CTDDConfig, device, rank=None):
        BackwardRate.__init__(self,cfg,device,rank)
        self.fix_logistic = cfg.model.fix_logistic

        tmp_net = load_temporal_network(config=cfg, device=device)

        if cfg.trainer.distributed:
            self.net = DDP(tmp_net, device_ids=[rank])
        else:
            self.net = tmp_net

        self.S = cfg.data1.vocab_size
        self.data_shape = cfg.data1.temporal_net_expected_shape
        self.device = device

    def _forward(self,
        x: TensorType["B", "D"],
        times: TensorType["B"]
    ) -> TensorType["B", "D", "S"]:
        """
            Returns logits over state space for each pixel
        """
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view(B, C*H*W)

        B, D = x.shape
        C,H,W = self.data_shape
        S = self.S
        x = x.view(B, C, H, W)

        net_out = self.net(x, times) # (B, 2*C, H, W)

        # Truncated logistic output from https://arxiv.org/pdf/2107.03006.pdf

        mu = net_out[:, 0:C, :, :].unsqueeze(-1)
        log_scale = net_out[:, C:, :, :].unsqueeze(-1)

        inv_scale = torch.exp(- (log_scale - 2))

        bin_width = 2. / self.S
        bin_centers = torch.linspace(start=-1. + bin_width/2,
            end=1. - bin_width/2,
            steps=self.S,
            device=self.device).view(1, 1, 1, 1, self.S)

        sig_in_left = (bin_centers - bin_width/2 - mu) * inv_scale
        bin_left_logcdf = F.logsigmoid(sig_in_left)
        sig_in_right = (bin_centers + bin_width/2 - mu) * inv_scale
        bin_right_logcdf = F.logsigmoid(sig_in_right)

        logits_1 = self._log_minus_exp(bin_right_logcdf, bin_left_logcdf)
        logits_2 = self._log_minus_exp(-sig_in_left + bin_left_logcdf, -sig_in_right + bin_right_logcdf)
        if self.fix_logistic:
            logits = torch.min(logits_1, logits_2)
        else:
            logits = logits_1

        logits = logits.view(B,D,S)

        return logits

    def _log_minus_exp(self, a, b, eps=1e-6):
        """
            Compute log (exp(a) - exp(b)) for (b<a)
            From https://arxiv.org/pdf/2107.03006.pdf
        """
        return a + torch.log1p(-torch.exp(b-a) + eps)

class BackRateMLP(EMA,BackwardRate,GaussianTargetRate):

    def __init__(self,config,device,rank=None):
        EMA.__init__(self,config)
        BackwardRate.__init__(self,config,device,rank)

        self.define_deep_models(device)
        self.init_ema()
        self.to(device)

    def define_deep_models(self,device):
        self.net = load_temporal_network(config=self.config,device=device)
        self.expected_temporal_output_shape = self.net.expected_output_shape
        if self.expected_temporal_output_shape != [self.dimensions,self.vocab_size]:
            temporal_output_total = reduce(lambda x, y: x * y, self.expected_temporal_output_shape)
            self.temporal_to_rate = nn.Linear(temporal_output_total,self.dimensions*self.vocab_size)

    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension", "num_states"]:

        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.view(B, C*H*W)

        x = self._center_data(x)
        batch_size = x.size(0)
        rate_logits = self.net(x,times)
        if self.net.expected_output_shape != [self.dimensions,self.vocab_size]:
            rate_logits = rate_logits.reshape(batch_size, -1)
            rate_logits = self.temporal_to_rate(rate_logits)
            rate_logits = rate_logits.reshape(batch_size,self.dimensions,self.vocab_size)
        return rate_logits

    def init_parameters(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

class BackRateConstant(EMA,BackwardRate,GaussianTargetRate):

    def __init__(self,config,device,rank=None,constant=10.):
        EMA.__init__(self,config)
        BackwardRate.__init__(self,config,device,rank)
        self.constant = constant
        self.define_deep_models()
        self.init_ema()


    def _forward(self,
                x: TensorType["batch_size", "dimension"],
                times: TensorType["batch_size"]
                ) -> TensorType["batch_size", "dimension", "num_states"]:

        x = self._center_data(x)
        batch_size = x.shape[0]

        return torch.full(torch.Size([batch_size, self.dimensions, self.vocab_size]), self.constant)



class GaussianTargetRateImageX0PredEMA(EMA,ImageX0PredBase):

    def __init__(self, cfg, device, rank=None):
        EMA.__init__(self, cfg)
        ImageX0PredBase.__init__(self, cfg, device, rank)
        self.config = cfg
        self.init_ema()


all_backward_rates = {"BackRateConstant":BackRateConstant,
                      "BackRateMLP":BackRateMLP,
                      "GaussianTargetRateImageX0PredEMA":GaussianTargetRateImageX0PredEMA}