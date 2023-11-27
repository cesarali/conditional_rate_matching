from dataclasses import dataclass

@dataclass
class DiffSamplerConfig:
    name:str = "DiffSampler"
    n_steps: int = 10
    approx: bool = False
    multi_hop: bool = False
    fixed_proposal: bool = False
    temp: float = 2.
    step_size: float = 1.0

@dataclass
class PerDimGibbsSamplerConfig:
    name:str = "PerDimGibbsSampler"
    rand:bool = False


