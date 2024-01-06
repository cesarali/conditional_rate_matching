from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    ConvNetAutoencoderConfig,
    TemporalDeepSetsConfig,
    TemporalGraphConvNetConfig,
    TemporalDeepMLPConfig,
    TemporalScoreNetworkAConfig,
    DiffusersUnet2DConfig
)

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (
    ConstantThermostatConfig,
    LogThermostatConfig
)

temporal_network_configs = {
    "TemporalMLP":TemporalMLPConfig,
    "ConvNetAutoencoder":ConvNetAutoencoderConfig,
    "TemporalDeepMLP":TemporalDeepMLPConfig,
    "TemporalDeepSets":TemporalDeepSetsConfig,
    "TemporalGraphConvNet":TemporalGraphConvNetConfig,
    "TemporalScoreNetworkA":TemporalScoreNetworkAConfig,
    "DiffusersUnet2D":DiffusersUnet2DConfig,
}

thermostat_configs = {
    "LogThermostat":LogThermostatConfig,
    "ConstantThermostat":ConstantThermostatConfig
}
