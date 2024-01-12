
from conditional_rate_matching.configs.configs_classes.config_dsb import DSBConfig
from .ctdd_reference import GaussianTargetRate
from .glauber_reference import GlauberDynamics


def load_reference(config:DSBConfig, device):
  if config.process.name == "GaussianTargetRate":
    process = GaussianTargetRate(config,device)
  elif config.process.name == "GlauberDynamics":
    process = GlauberDynamics(config,device)
  else:
    raise Exception("No Reference Defined")

  return process
