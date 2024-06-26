from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.models.temporal_networks.temporal_graphs import TemporalGraphConvNet
from conditional_rate_matching.models.temporal_networks.temporal_convnet import UConvNISTNet
from conditional_rate_matching.models.temporal_networks.temporal_mlp import TemporalDeepMLP, TemporalLeNet5, TemporalUNet, TemporalLeNet5Autoencoder, TemporalMLP

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    TemporalLeNet5Config,
    TemporalLeNet5AutoencoderConfig,
    TemporalUNetConfig,
    UConvNISTNetConfig,
    TemporalGraphConvNetConfig,
    TemporalDeepMLPConfig,
    SequenceTransformerConfig,
    SimpleTemporalGCNConfig
)

from conditional_rate_matching.models.temporal_networks.temporal_diffusers_wrappers import DiffusersUnet2D
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import DiffusersUnet2DConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import CFMUnetConfig

from conditional_rate_matching.models.temporal_networks.temporal_gnn.TemporalScoreNetwork_A import TemporalScoreNetworkA
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig
from conditional_rate_matching.models.temporal_networks.temporal_transformers import SequenceTransformer

from conditional_rate_matching.models.temporal_networks.unet import UNetModelWrapper
from conditional_rate_matching.models.temporal_networks.temporal_gnn.geometric_gnn import SimpleTemporalGCN

def load_temporal_network(config:CRMConfig, device):
    if isinstance(config.temporal_network,TemporalMLPConfig):
        temporal_network = TemporalMLP(config,device)
    elif isinstance(config.temporal_network,TemporalDeepMLPConfig):
        temporal_network = TemporalDeepMLP(config,device)
    elif isinstance(config.temporal_network,TemporalLeNet5Config):
        temporal_network = TemporalLeNet5(config,device)
    elif isinstance(config.temporal_network,TemporalLeNet5AutoencoderConfig):
        temporal_network = TemporalLeNet5Autoencoder(config,device)
    elif isinstance(config.temporal_network,TemporalUNetConfig):
        temporal_network = TemporalUNet(config,device)
    elif isinstance(config.temporal_network,UConvNISTNetConfig):
        temporal_network = UConvNISTNet(config)
        temporal_network = temporal_network.to(device)
    elif isinstance(config.temporal_network,TemporalGraphConvNetConfig):
        temporal_network = TemporalGraphConvNet(config, device)
    elif isinstance(config.temporal_network, DiffusersUnet2DConfig):
        temporal_network = DiffusersUnet2D(config,device)
    elif isinstance(config.temporal_network,TemporalScoreNetworkAConfig):
        temporal_network = TemporalScoreNetworkA(config)
        temporal_network = temporal_network.to(device)
    elif isinstance(config.temporal_network, TemporalLeNet5Config):
        temporal_network = TemporalLeNet5(config, device)
    elif isinstance(config.temporal_network, SequenceTransformerConfig):
        temporal_network = SequenceTransformer(config, device)
    elif isinstance(config.temporal_network,CFMUnetConfig):
        temporal_network = UNetModelWrapper(**config.temporal_network.__dict__).to(device)
    elif isinstance(config.temporal_network,SimpleTemporalGCNConfig):
        temporal_network = SimpleTemporalGCN(config).to(device)
    else:
        raise Exception("Temporal Network not Defined")
    return temporal_network