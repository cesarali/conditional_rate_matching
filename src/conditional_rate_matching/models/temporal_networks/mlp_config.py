from dataclasses import dataclass

@dataclass
class ResNetEBMConfig:
    name:str = "ResNetEBM"
    n_channels:int = 64
    n_blocks:int = 6