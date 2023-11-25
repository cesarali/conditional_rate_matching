import unittest
from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles

import torch
from conditional_rate_matching.configs.config_oops import OopsConfig

from conditional_rate_matching.data.graph_dataloaders_config import CommunitySmallConfig

from conditional_rate_matching.models.temporal_networks.ebm import EBM
from conditional_rate_matching.models.temporal_networks.mlp_utils import get_net
from conditional_rate_matching.models.temporal_networks.mlp_config import MLPEBMConfig,ResNetEBMConfig

from conditional_rate_matching.models.generative_models.oops import Oops
from conditional_rate_matching.data.dataloaders_utils import get_dataloader_oops
from conditional_rate_matching.models.pipelines.mc_samplers.oops_samplers_utils import get_oops_samplers

from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import (
    PerDimGibbsSamplerConfig,
    DiffSamplerConfig,
)

class TestPipeline(unittest.TestCase):

    def test_sampler_gibbs(self):
        experiment_files = ExperimentFiles(experiment_name="oops",
                                           experiment_type="mnist",
                                           experiment_indentifier="test",
                                           delete=True)

        configs = OopsConfig()
        configs.data0 = CommunitySmallConfig(flatten=True,as_image=False,full_adjacency=True)
        configs.model_mlp = MLPEBMConfig(hidden_size=19)

        configs.sampler = PerDimGibbsSamplerConfig()

        device = torch.device(configs.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        oops = Oops(config=configs,device=device,experiment_files=experiment_files)
        oops.pipeline.initialize(device)

        x,ll = oops.pipeline(oops.model,sample_size=23,return_path=False,get_ll=True)
        print(x.shape)




if __name__=="__main__":
    unittest.main()