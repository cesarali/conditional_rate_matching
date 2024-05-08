import torch
from torch import nn
from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
from conditional_rate_matching.models.networks.mlp_utils import get_net


class EBM(nn.Module):
    def __init__(self, config:OopsConfig, mean=None,device=torch.device("cpu")):

        super().__init__()
        self.net = get_net(config,device)
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            base_dist = torch.distributions.Bernoulli(probs=self.mean)
            bd = base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp + bd