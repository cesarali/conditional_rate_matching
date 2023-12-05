from dataclasses import dataclass

@dataclass
class ConstantThermostatConfig:
    name:str="ConstantThermostat"
    gamma:float = .9

@dataclass
class LogThermostatConfig:
    name:str="LogThermostat"
    time_exponential:float = 3.
    time_base:float = 1.0