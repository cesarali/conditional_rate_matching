from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.models.networks.mlp_config import MLPConfig
from conditional_rate_matching.models.networks.mlp import MLP
def get_conditional_network(config:CRMConfig,device):
    if isinstance(config.conditional_network,MLPConfig):
        conditional_network = MLP(config.conditional_network)
        conditional_network = conditional_network.to(device)
    else:
        raise Exception("No conditional network defined")

    return conditional_network