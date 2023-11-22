import os
from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.models.temporal_networks.mlp import ResNetEBM
from conditional_rate_matching.models.temporal_networks.mlp_config import ResNetEBMConfig

def get_net(config:OopsConfig, device):
    if isinstance(config.model_mlp,ResNetEBMConfig):
        mlp = ResNetEBM(config).to(device)
    else:
        raise Exception("No Network")
    return mlp