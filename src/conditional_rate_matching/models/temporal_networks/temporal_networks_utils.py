from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.models.temporal_networks.temporal_graphs import TemporalGraphConvNet
from conditional_rate_matching.models.temporal_networks.temporal_convnet import UConvNISTNet
from conditional_rate_matching.models.temporal_networks.temporal_mlp import TemporalDeepMLP, TemporalLeNet5, TemporalUNet, TemporalLeNet5Autoencoder, TemporalMLP

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalDeepMLPConfig,
    ConvNetAutoencoderConfig
)

from conditional_rate_matching.models.temporal_networks.mlp import (
    TemporalDeepMLP,
    ConvNetAutoencoder
)

def load_temporal_network(config:CRMConfig, device):
    if isinstance(config.temporal_network,TemporalDeepMLPConfig):
        temporal_network = TemporalDeepMLP(config,device)
    elif isinstance(config.temporal_network,ConvNetAutoencoderConfig):
        temporal_network = ConvNetAutoencoder(config,device)
    else:
        raise Exception("Temporal Network not Defined")

    return temporal_network
