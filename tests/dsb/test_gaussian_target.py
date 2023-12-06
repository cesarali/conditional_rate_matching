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

def test_glauber_from_gm():
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="test",
                                           experiment_indentifier="test")
    config = experiment_comunity_small()
    pprint(config)
    dsb = DSB(config,experiment_files=experiment_files)


    databatch1 = next(dsb.dataloader_1.train().__iter__())
    x1 = databatch1[0].to(dsb.device)
    time_grid = dsb.pipeline.get_time_steps()
    path, time_grid = dsb.process.sample_path(x1,time_grid)