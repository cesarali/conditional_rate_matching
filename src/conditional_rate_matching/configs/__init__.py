from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    ConvNetAutoencoderConfig,
    TemporalDeepSetsConfig,
    TemporalGraphConvNetConfig,
    TemporalDeepMLPConfig,
    TemporalDeepEBMConfig
)

temporal_network_configs = {
    "TemporalMLP":TemporalMLPConfig,
    "ConvNetAutoencoder":ConvNetAutoencoderConfig,
    "TemporalDeepMLP":TemporalDeepMLPConfig,
    "TemporalDeepSets":TemporalDeepSetsConfig,
    "TemporalGraphConvNet":TemporalGraphConvNetConfig,
    "TemporalDeepMLP":TemporalDeepEBMConfig
}
