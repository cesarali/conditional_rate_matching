from conditional_rate_matching.configs.config_crm import Config
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig
from conditional_rate_matching.models.temporal_networks.mlp import TemporalMLP

def load_temporal_network(config:Config,device):
    if isinstance(config.temporal_network,TemporalMLPConfig):
        temporal_network = TemporalMLP(config,device)
    else:
        raise Exception("Temporal Network not Defined")

    return temporal_network