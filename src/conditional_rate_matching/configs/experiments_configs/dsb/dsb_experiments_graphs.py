import os
from pprint import pprint
from dataclasses import asdict

from conditional_rate_matching.configs.config_dsb import DSBConfig

from conditional_rate_matching.data.graph_dataloaders_config import (
    EgoConfig,
    GridConfig,
    CommunitySmallConfig
)

from conditional_rate_matching.models.losses.dsb_losses_config import SteinSpinEstimatorConfig
from conditional_rate_matching.configs.config_dsb import DSBTrainerConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.models.pipelines.reference_process.reference_process_config import (
    GlauberDynamicsConfig,
    GaussianTargetRateConfig
)
from conditional_rate_matching.data.ctdd_target_config import GaussianTargetConfig

def experiment_comunity_small(number_of_epochs=300,berlin=True):
    crm_config = DSBConfig()
    crm_config.data0 = CommunitySmallConfig(flatten=True,as_image=False,full_adjacency=False,batch_size=20)
    crm_config.data1 = GaussianTargetConfig()
    crm_config.process = GaussianTargetRateConfig()

    crm_config.pipeline.number_of_steps = 25
    crm_config.temporal_network.hidden_dim = 100
    crm_config.temporal_network.time_embed_dim = 100
    crm_config.flip_estimator = SteinSpinEstimatorConfig()

    crm_config.trainer = DSBTrainerConfig(number_of_epochs=number_of_epochs,
                                          berlin=berlin,
                                          metrics=[],
                                          learning_rate=1e-5,
                                          do_ema=False)
    return crm_config

