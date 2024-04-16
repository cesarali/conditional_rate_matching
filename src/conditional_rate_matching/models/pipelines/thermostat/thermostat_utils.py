
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat import ConstantThermostat,LogThermostat
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat import ExponentialThermostat
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat import InvertedExponentialThermostat
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat import PeriodicThermostat
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat import PolynomialThermostat
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat import PlateauThermostat

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import LogThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ExponentialThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import InvertedExponentialThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import PeriodicThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import PolynomialThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import PlateauThermostatConfig

def load_thermostat(config:CRMConfig):
  if isinstance(config.thermostat,ConstantThermostatConfig):
    thermostat = ConstantThermostat(config.thermostat)
  elif isinstance(config.thermostat,LogThermostatConfig):
    thermostat = LogThermostat(config.thermostat)
  elif isinstance(config.thermostat,ExponentialThermostatConfig):
    thermostat = ExponentialThermostat(config.thermostat)
  elif isinstance(config.thermostat, InvertedExponentialThermostatConfig):
    thermostat = InvertedExponentialThermostat(config.thermostat)
  elif isinstance(config.thermostat, PeriodicThermostatConfig):
    thermostat = PeriodicThermostat(config.thermostat)
  elif isinstance(config.thermostat, PolynomialThermostatConfig):
    thermostat = PolynomialThermostat(config.thermostat)
  elif isinstance(config.thermostat, PlateauThermostatConfig):
    thermostat = PlateauThermostat(config.thermostat)
  else:
    raise Exception("No Thermostat Defined")
  return thermostat
