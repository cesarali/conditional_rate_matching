import pytest
import torch
from conditional_rate_matching.configs.config_dsb import DSBConfig

from pprint import pprint
from dataclasses import asdict
from conditional_rate_matching.models.pipelines.reference_process.reference_process_utils import load_reference
from conditional_rate_matching.models.generative_models.spin_glass.spin_glasses_configs import SpinGlassVariablesConfig
from conditional_rate_matching.models.generative_models.spin_glass.spin_glasses_parametrized import ParametrizedSpinGlassHamiltonian

import os
import torch
import pytest
from conditional_rate_matching.configs.experiments_configs.dsb.dsb_experiments_graphs import experiment_comunity_small

from conditional_rate_matching.configs.config_dsb import DSBConfig
from conditional_rate_matching.models.generative_models.dsb import DSB
from conditional_rate_matching.models.generative_models.dsb import DSBExperimentsFiles

def test_load_model():
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="graph",
                                           experiment_indentifier="training_test",
                                           delete=True)
    dsb = DSB(experiment_dir=experiment_files.experiment_dir,sinkhorn_iteration=4)
