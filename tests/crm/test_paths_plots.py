import os
import torch
from conditional_rate_matching.models.generative_models.crm import CRM

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping


def test_bridge_rate():
    config : CRMConfig
    config = experiment_nist(dataset_name="mnist",temporal_network_name="mlp")
    crm = CRM(config)

    databatch_0 = next(crm.dataloader_0.train().__iter__())
    x_0 = databatch_0[0]

    databatch_1 = next(crm.dataloader_1.train().__iter__())
    x_1 = databatch_1[0]

    # rate_model = lambda x, t: constant_rate(config, x, t)
    rate_model = lambda x, t: crm.forward_rate.conditional_transition_rate(x, x_1, t)
    x_f, x_hist, x0_hist, ts = TauLeaping(config, rate_model, x_0, forward=True)
    print(x_hist.shape)

def test_variance_bridge():
    config : CRMConfig
    config = experiment_nist(dataset_name="mnist",temporal_network_name="mlp")
    crm = CRM(config)

    databatch_0 = next(crm.dataloader_0.train().__iter__())
    x_0 = databatch_0[0]

    databatch_1 = next(crm.dataloader_1.train().__iter__())
    x_1 = databatch_1[0]

    t = torch.rand((x_0.size(0)))

    variance = crm.forward_rate.compute_variance_torch(t,x_1,x_0)
    print(variance)



