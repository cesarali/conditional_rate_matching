from conditional_rate_matching.configs.config_crm import CRMConfig

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    TemporalDeepMLPConfig,
    TemporalDeepSetsConfig,
    TemporalGNNConfig,
    ConvNetAutoencoderConfig
)

from conditional_rate_matching.models.temporal_networks.mlp import (
    TemporalMLP,
    TemporalDeepMLP,
    TemporalDeepSets,
    TemporalGNN,
    ConvNetAutoencoder
)

def load_temporal_network(config:CRMConfig, device):
    if isinstance(config.temporal_network,TemporalMLPConfig):
        temporal_network = TemporalMLP(config,device)
    elif isinstance(config.temporal_network,TemporalDeepMLPConfig):
        temporal_network = TemporalDeepMLP(config,device)
    elif isinstance(config.temporal_network,TemporalDeepSetsConfig):
        temporal_network = TemporalDeepSets(config,device)
    elif isinstance(config.temporal_network,TemporalGNNConfig):
        temporal_network = TemporalGNN(config,device)
    elif isinstance(config.temporal_network,ConvNetAutoencoderConfig):
        temporal_network = ConvNetAutoencoder(config,device)
    else:
        raise Exception("Temporal Network not Defined")

    return temporal_network