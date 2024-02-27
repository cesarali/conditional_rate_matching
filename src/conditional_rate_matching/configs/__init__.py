from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    TemporalLeNet5Config,
    TemporalLeNet5AutoencoderConfig,
    TemporalUNetConfig,
    ConvNetAutoencoderConfig,
    TemporalGraphConvNetConfig,
    TemporalDeepMLPConfig,
    TemporalScoreNetworkAConfig,
    SequenceTransformerConfig,
    DiffusersUnet2DConfig,
    UConvNISTNetConfig
)

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (
    ConstantThermostatConfig,
    LogThermostatConfig,
    ExponentialThermostatConfig,
    InvertedExponentialThermostatConfig
)

from conditional_rate_matching.models.networks.mlp_config import MLPConfig

conditional_network_configs = {
    "MLP":MLPConfig
}

temporal_network_configs = {
    "TemporalMLP":TemporalMLPConfig,
    "TemporalLeNet5":TemporalLeNet5Config,
    "ConvNetAutoencoder":ConvNetAutoencoderConfig,
    "SequenceTransformer":SequenceTransformerConfig,
    "TemporalDeepMLP":TemporalDeepMLPConfig,
    "TemporalLeNet5":TemporalLeNet5Config,
    "TemporalLeNet5Autoencoder":TemporalLeNet5AutoencoderConfig,
    "TemporalUNet":TemporalUNetConfig,
    "TemporalGraphConvNet":TemporalGraphConvNetConfig,
    "TemporalScoreNetworkA":TemporalScoreNetworkAConfig,
    "DiffusersUnet2D":DiffusersUnet2DConfig,
    "UConvNISTNet":UConvNISTNetConfig
}

thermostat_configs = {
    "LogThermostat":LogThermostatConfig,
    "ConstantThermostat":ConstantThermostatConfig,
    "ExponentialThermostat":ExponentialThermostatConfig,
    "InvertedExponentialThermostat":InvertedExponentialThermostatConfig
}
