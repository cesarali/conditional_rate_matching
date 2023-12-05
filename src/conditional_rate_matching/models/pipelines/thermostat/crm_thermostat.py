import math
import torch
from torchtyping import TensorType

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import LogThermostatConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig


class ConstantThermostat:

    def __init__(self,config:ConstantThermostatConfig):
        self.gamma = config.gamma

    def __call__(self, t):
        device = t.device
        thermostat = torch.full_like(t,self.gamma).to(device)
        return thermostat

    def integral(self,t0,t1):
        interval = t1 - t0
        integral = self.gamma * interval
        return integral


class LogThermostat:

    def __init__(self,config:LogThermostatConfig):
        self.time_base = config.time_base
        self.time_exponential = config.time_exponential

    def _integral_rate_scalar(self, t: TensorType["B"]) -> TensorType["B"]:
        integral_ = self.time_base * (self.time_exponential ** t) - self.time_base
        return integral_

    def __call__(self, t: TensorType["B"]) -> TensorType["B"]:
        device = t.device
        thermostat = self.time_base * math.log(self.time_exponential)* (self.time_exponential ** (1.- t))
        return thermostat.to(device)


