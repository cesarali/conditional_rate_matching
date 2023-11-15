from conditional_rate_matching.configs.config_crm import Config

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    ConvNetAutoencoderConfig
)

from conditional_rate_matching.models.temporal_networks.mlp import (
    TemporalMLP,
    ConvNetAutoencoder
)

def load_temporal_network(config:Config,device):
    if isinstance(config.temporal_network,TemporalMLPConfig):
        temporal_network = TemporalMLP(config,device)
    elif isinstance(config.temporal_network,ConvNetAutoencoderConfig):
        temporal_network = ConvNetAutoencoder(config,device)
    else:
        raise Exception("Temporal Network not Defined")

    return temporal_network