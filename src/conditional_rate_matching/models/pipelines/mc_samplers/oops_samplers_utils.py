from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
from conditional_rate_matching.models.pipelines.mc_samplers.oops_samplers import PerDimGibbsSampler,DiffSampler
from conditional_rate_matching.models.pipelines.mc_samplers.oops_sampler_config import DiffSamplerConfig,PerDimGibbsSamplerConfig

def get_oops_samplers(config:OopsConfig):
    if isinstance(config.sampler,DiffSamplerConfig):
        sampler = DiffSampler(config)
    elif isinstance(config.sampler,PerDimGibbsSamplerConfig):
        sampler = PerDimGibbsSampler(config)
    else:
        raise Exception("Sampler not Implemented")
    return sampler