import os
from pprint import pprint
from dataclasses import asdict

from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.models.temporal_networks.mlp_config import ResNetEBMConfig,MLPEBMConfig
from conditional_rate_matching.data.graph_dataloaders_config import EgoConfig,GridConfig,CommunitySmallConfig
from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import DiffSamplerConfig,PerDimGibbsSamplerConfig

def experiment_ego(gibbs=True,berlin=True):
    oops_config = OopsConfig()
    oops_config.data0 = EgoConfig(flatten=True,as_image=False,full_adjacency=False)
    oops_config.model_mlp = MLPEBMConfig()
    if gibbs:
        oops_config.sampler = PerDimGibbsSamplerConfig()
    else:
        oops_config.sampler = DiffSamplerConfig()
    oops_config.trainer.berlin = berlin
    return oops_config

def experiment_comunity_small(gibbs=True,berlin=True):
    oops_config = OopsConfig()
    oops_config.data0 = CommunitySmallConfig(flatten=True,as_image=False,full_adjacency=False)
    oops_config.model_mlp = MLPEBMConfig()
    if gibbs:
        oops_config.sampler = PerDimGibbsSamplerConfig()
    else:
        oops_config.sampler = DiffSamplerConfig()
    oops_config.trainer.berlin = berlin
    return oops_config

def experiment_grid(gibbs=True,berlin=True):
    oops_config = OopsConfig()
    oops_config.data0 = GridConfig(flatten=True,as_image=False,full_adjacency=False)
    oops_config.model_mlp = MLPEBMConfig()
    if gibbs:
        oops_config.sampler = PerDimGibbsSamplerConfig()
    else:
        oops_config.sampler = DiffSamplerConfig()
    oops_config.trainer.berlin = berlin
    return oops_config


if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    #config = experiment_grid(gibbs=True,berlin=True)
    config = experiment_comunity_small(gibbs=False,berlin=True)
    #config = experiment_ego(gibbs=True,berlin=True)
    config.data0.batch_size = 128

    config.sampler.n_steps = 1
    config.pipeline.number_of_betas = 10

    config.trainer.debug = True
    config.trainer.number_of_epochs = 4
    config.trainer.max_test_size = 2

    pprint(asdict(config))
    call_trainer(config)