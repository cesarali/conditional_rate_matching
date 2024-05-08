import unittest

import torch

from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
from conditional_rate_matching.models.networks.ebm import EBM
from conditional_rate_matching.models.networks.mlp_config import ResNetEBMConfig


from conditional_rate_matching.data.dataloaders_utils import get_dataloader_oops
from conditional_rate_matching.models.pipelines.mc_samplers.oops_samplers_utils import get_oops_samplers

from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import (
    PerDimGibbsSamplerConfig,
    DiffSamplerConfig,
)

class TestSampler(unittest.TestCase):

    @unittest.skip
    def test_sampler_gibbs(self):
        configs = OopsConfig()
        configs.sampler = PerDimGibbsSamplerConfig()
        device = torch.device(configs.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        dataloader = get_dataloader_oops(configs)
        databatch = next(dataloader.train().__iter__())
        x = databatch[0].to(device)
        ebm = EBM(configs,device=device).to(device)
        sampler = get_oops_samplers(configs,device)
        sample = sampler.step(x,ebm)
        print(sample.device)
        print(sample.shape)

    def test_diff(self):
        configs = OopsConfig()
        configs.model_mlp = ResNetEBMConfig(n_blocks=1,
                                            n_channels=2)

        configs.sampler = DiffSamplerConfig()
        device = torch.device(configs.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        dataloader = get_dataloader_oops(configs)
        databatch = next(dataloader.train().__iter__())
        x = databatch[0].to(device)
        ebm = EBM(configs,device=device).to(device)
        sampler = get_oops_samplers(configs)
        sample = sampler.step(x,ebm)
        print(sample.device)
        print(sample.shape)


if __name__=="__main__":
    unittest.main()