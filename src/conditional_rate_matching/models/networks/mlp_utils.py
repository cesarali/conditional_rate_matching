from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
from conditional_rate_matching.models.networks.mlp import ResNetEBM
from conditional_rate_matching.models.networks.mlp import MLP_EBM
from conditional_rate_matching.models.networks.mlp_config import ResNetEBMConfig
from conditional_rate_matching.models.networks.mlp_config import MLPEBMConfig


def get_net(config:OopsConfig, device):
    if isinstance(config.model_mlp,ResNetEBMConfig):
        mlp = ResNetEBM(config).to(device)
    elif isinstance(config.model_mlp,MLPEBMConfig):
        mlp = MLP_EBM(config).to(device)
    else:
        raise Exception("No Network")
    return mlp