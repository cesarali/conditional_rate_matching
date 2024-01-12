from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.models.temporal_networks.temporal_graphs import TemporalGraphConvNet
from conditional_rate_matching.models.temporal_networks.temporal_convnet import UConvNISTNet
from conditional_rate_matching.models.temporal_networks.temporal_deep_set import TemporalDeepSets
from conditional_rate_matching.models.temporal_networks.temporal_mlp import TemporalMLP
from conditional_rate_matching.models.temporal_networks.temporal_mlp import TemporalDeepMLP

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    UConvNISTNetConfig,
    TemporalDeepSetsConfig,
    TemporalGraphConvNetConfig,
    TemporalDeepMLPConfig
)

from conditional_rate_matching.models.temporal_networks.temporal_diffusers_wrappers import DiffusersUnet2D
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import DiffusersUnet2DConfig

from conditional_rate_matching.models.temporal_networks.temporal_gnn.TemporalScoreNetwork_A import TemporalScoreNetworkA
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig


def load_temporal_network(config:CRMConfig, device):
    if isinstance(config.temporal_network,TemporalMLPConfig):
        temporal_network = TemporalMLP(config,device)
    if isinstance(config.temporal_network,TemporalDeepMLPConfig):
        temporal_network = TemporalDeepMLP(config,device)
    elif isinstance(config.temporal_network,UConvNISTNetConfig):
        temporal_network = UConvNISTNet(config)
        temporal_network = temporal_network.to(device)
    elif isinstance(config.temporal_network,TemporalGraphConvNetConfig):
        temporal_network = TemporalGraphConvNet(config, device)
    elif isinstance(config.temporal_network,TemporalDeepSetsConfig):
        temporal_network = TemporalDeepSets(config, device)
    elif isinstance(config.temporal_network, DiffusersUnet2DConfig):
        temporal_network = DiffusersUnet2D(config,device)
    elif isinstance(config.temporal_network,TemporalScoreNetworkAConfig):
        temporal_network = TemporalScoreNetworkA(config)
        temporal_network = temporal_network.to(device)
    else:
        raise Exception("Temporal Network not Defined")
    return temporal_network