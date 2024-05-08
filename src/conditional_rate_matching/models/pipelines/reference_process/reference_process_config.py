import os
import torch
from pathlib import Path
from typing import List,Union
from dataclasses import dataclass
from conditional_rate_matching.models.generative_models.spin_glass.spin_glasses_configs import SpinGlassVariablesConfig

@dataclass
class GaussianTargetRateConfig:
    """
    Reference configuration for schrodinger bridge reference process
    """
    # reference process variables
    name:str = "GaussianTargetRate"
    initial_dist:str = 'gaussian'
    rate_sigma:float = 6.0
    Q_sigma:float = 512.0
    time_exponential:float = 3.
    time_base:float = 1.0


@dataclass
class GlauberDynamicsConfig(SpinGlassVariablesConfig):

    name:str = "GlauberDynamics"
    fom_data_hamiltonian:bool = True #defines if the fields and coupling depend on the data

    beta:float=1.
    gamma:float = 1.
    fields:List[float] = None
    couplings:List[float] = None


all_reference_process_configs = {"GaussianTargetRate":GaussianTargetRateConfig,
                                 "GlauberDynamics":GlauberDynamicsConfig}