from conditional_rate_matching.configs.config_files import ExperimentFiles
import unittest

import torch
from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.models.generative_models.oops import Oops
from conditional_rate_matching.models.metrics.metrics_utils import log_metrics

from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import (
    PerDimGibbsSamplerConfig,
)

class TestOops(unittest.TestCase):

    def test_oops(self):
        experiment_files = ExperimentFiles(experiment_name="oops",
                                           experiment_type="mnist",
                                           experiment_indentifier="test",
                                           delete=True)
        experiment_files.create_directories()
        configs = OopsConfig()
        configs.model_mlp.n_blocks = 1
        configs.model_mlp.n_channels = 1

        configs.pipeline.number_of_betas = 10
        configs.pipeline.sampler_steps_per_ais_iter = 1

        configs.sampler = PerDimGibbsSamplerConfig()
        device = torch.device(configs.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        oops = Oops(config=configs,device=device,experiment_files=experiment_files)
        #x,ll = oops.pipeline(oops.model,23,return_path=False,get_ll=True)
        #print(x.shape)

        all_metrics = log_metrics(oops,0,metrics_to_log=[MetricsAvaliable.mse_histograms,MetricsAvaliable.mnist_plot])

        #print(x.shape)


