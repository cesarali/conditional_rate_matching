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

@dataclass
class ExponentialThermostatConfig:
    name:str="ExponentialThermostat"
    max:float = 10.
    gamma:float = 10.

@dataclass
class InvertedExponentialThermostatConfig:
    name:str="InvertedExponentialThermostat"
    max:float = 10.
    gamma:float = 10.
