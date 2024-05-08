from typing import List
from dataclasses import dataclass,field,fields


@dataclass
class MLPConfig:
    name:str = "MLP"
    input_dim:int = 2
    layers_dim: List[int] = field(default_factory=lambda:[150,150])
    dropout:float = 0.2
    ouput_dim: int = 3
    normalization: bool = True
    ouput_transformation:str = "relu" #sigmoid


@dataclass
class ResNetEBMConfig:
    name:str = "ResNetEBM"
    n_channels:int = 64
    n_blocks:int = 6

@dataclass
class MLPEBMConfig:
    name:str = "MLPEBM"
    hidden_size:int = 256
