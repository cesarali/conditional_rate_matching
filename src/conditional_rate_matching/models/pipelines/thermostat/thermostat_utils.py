
from conditional_rate_matching.configs.config_crm import CRMConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat import ConstantThermostat,LogThermostat

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import LogThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig


def load_thermostat(config:CRMConfig):
  if isinstance(config.thermostat,ConstantThermostatConfig):
    thermostat = ConstantThermostat(config.thermostat)
  elif isinstance(config.thermostat,LogThermostatConfig):
    thermostat = LogThermostat(config.thermostat)
  else:
    raise Exception("No Thermostat Defined")

  return thermostat
