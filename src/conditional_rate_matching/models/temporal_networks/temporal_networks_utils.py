from conditional_rate_matching.configs.config_crm import CRMConfig

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    ConvNetAutoencoderConfig,
    TemporalDeepMLPConfig,
    TemporalDeepEBMConfig,
    TemporalDeepSetsConfig,
    TemporalGraphConvNetConfig
)

from conditional_rate_matching.models.temporal_networks.temporal_convnet import ConvNetAutoencoder
from conditional_rate_matching.models.temporal_networks.temporal_mlp import TemporalMLP, TemporalDeepMLP
from conditional_rate_matching.models.temporal_networks.temporal_EBM import TemporalDeepEBM
from conditional_rate_matching.models.temporal_networks.temporal_deep_set import TemporalDeepSets
from conditional_rate_matching.models.temporal_networks.temporal_graphs import TemporalGraphConvNet


def load_temporal_network(config:CRMConfig, device):
    if isinstance(config.temporal_network,TemporalMLPConfig):
        temporal_network = TemporalMLP(config,device)
    elif isinstance(config.temporal_network,TemporalDeepMLPConfig):
        temporal_network = TemporalDeepMLP(config,device)
    elif isinstance(config.temporal_network,TemporalDeepEBMConfig):
        temporal_network = TemporalDeepEBM(config,device)
    elif isinstance(config.temporal_network,TemporalDeepSetsConfig):
        temporal_network = TemporalDeepSets(config,device)
    elif isinstance(config.temporal_network,TemporalGraphConvNetConfig):
        temporal_network = TemporalGraphConvNet(config,device)
    elif isinstance(config.temporal_network,ConvNetAutoencoderConfig):
        temporal_network = ConvNetAutoencoder(config,device)
    elif isinstance(config.temporal_network,TemporalGraphConvNetConfig):
        temporal_network = TemporalGraphConvNet(config, device)
    elif isinstance(config.temporal_network,TemporalDeepSetsConfig):
        temporal_network = TemporalDeepSets(config, device)
    else:
        raise Exception("Temporal Network not Defined")

    return temporal_network