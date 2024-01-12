from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (
    TemporalMLPConfig,
    ConvNetAutoencoderConfig,
    TemporalDeepSetsConfig,
    TemporalGraphConvNetConfig,
    TemporalDeepMLPConfig,
<<<<<<< HEAD
    TemporalDeepEBMConfig
=======
    TemporalScoreNetworkAConfig,
    DiffusersUnet2DConfig,
    UConvNISTNetConfig
>>>>>>> origin/main
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
<<<<<<< HEAD
    "TemporalDeepMLP":TemporalDeepEBMConfig
=======
    "TemporalScoreNetworkA":TemporalScoreNetworkAConfig,
    "DiffusersUnet2D":DiffusersUnet2DConfig,
    "UConvNISTNet":UConvNISTNetConfig
>>>>>>> origin/main
}

thermostat_configs = {
    "LogThermostat":LogThermostatConfig,
    "ConstantThermostat":ConstantThermostatConfig
}
