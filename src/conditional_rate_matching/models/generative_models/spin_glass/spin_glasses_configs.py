import os
from typing import List
from dataclasses import dataclass,field

@dataclass
class SpinGlassVariablesConfig:
    #ISING VARIABLES
    number_of_spins: int = 4
    beta: int = 1.
    obtain_partition_function: bool = True
    couplings_deterministic: float = 1.
    couplings_sigma: float = 1.
    couplings: List[float] = None
    fields: List[float] = None
    mcmc_sample_size: int = 800
    number_of_mcmc_steps: int = 1000
    number_of_mcmc_burning_steps: int = 500
