from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    TemporalLeNet5Config,
    TemporalLeNet5AutoencoderConfig,
    TemporalUNetConfig,
    ConvNetAutoencoderConfig,
    TemporalGraphConvNetConfig,
    TemporalDeepMLPConfig,
    TemporalScoreNetworkAConfig,
    DiffusersUnet2DConfig,
    UConvNISTNetConfig
)

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (
    ConstantThermostatConfig,
    LogThermostatConfig,
    ExponentialThermostatConfig,
    InvertedExponentialThermostatConfig
)

temporal_network_configs = {
    "TemporalMLP":TemporalMLPConfig,
    "ConvNetAutoencoder":ConvNetAutoencoderConfig,
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
