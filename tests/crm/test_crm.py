import torch
import numpy as np
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig as ConditionalRateMatchingConfig

from conditional_rate_matching.models.generative_models.crm import uniform_pair_x0_x1
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm

from conditional_rate_matching.models.generative_models.crm import (
    CRM
)

from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.config_files import get_experiment_dir
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
import pytest

def test_training():
    experiment_files = ExperimentFiles(experiment_name="crm",experiment_type="trainer_call")
    config = CRMConfig()
    config.optimal_transport.name = "uniform"
    trainer = CRMTrainer(config,experiment_files=experiment_files)
    trainer.train()

@pytest.mark.skip
def test_optimal_trasnport():
    seed = 1980
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="trainer_call")
    config = CRMConfig()
    crm = CRM(config,experiment_files=experiment_files)
    batch_1, batch_0 = next(zip(crm.dataloader_1.train(), crm.dataloader_0.train()).__iter__())
    x1 = batch_1[0]
    x0 = batch_0[0]
    #batch_size = x1.size(0)
    batch_size = 23
    torch.manual_seed(seed)
    np.random.seed(seed)


    pi = crm.op_sampler.get_map(x0, x1)
    indices_i, indices_j = crm.op_sampler.sample_map(pi, batch_size=batch_size, replace=True)
    new_x0, new_x1 = x0[indices_i], x1[indices_j]

    torch.manual_seed(seed)
    np.random.seed(seed)

    sampled_x0, sampled_x1 = crm.op_sampler.sample_plan(x0, x1, replace=True)

    assert torch.equal(new_x0, sampled_x0)
    assert torch.equal(new_x1, sampled_x1)

def test_conditional_probability():
    config = ConditionalRateMatchingConfig()
    dataloader_0,dataloader_1 = get_dataloaders_crm(config)

    batch_1, batch_0 = next(zip(dataloader_1.train(), dataloader_0.train()).__iter__())
    x_0 = batch_0[0]
    x_1 = batch_1[0]
    time = torch.rand((x_0.size(0)))
    x_1,x_0 = uniform_pair_x0_x1(batch_1,batch_0)

    where_to_x = torch.arange(0, config.vocab_size)
    where_to_x = where_to_x[None, None, :].repeat((x_0.size(0), config.dimensions, 1)).float()
    where_to_x = where_to_x.to(x_0.device)

    probs = conditional_probability(config, where_to_x, x_0, time, t0=0.)
    probs_transition = telegram_bridge_probability(config, where_to_x, x_1, x_0, time)

    print(probs.sum(axis=-1))
    print(probs_transition.sum(axis=-1))

#@pytest.mark.skip(reason="No way of currently testing this")
def test_load():

    #experiment_dir = get_experiment_dir(experiment_name="ot_test",
    #                                    experiment_type="trainer_call",
    #                                    experiment_indentifier="1701864662")
    experiment_dir = "C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/conditional_rate_matching/results/emnist_to_mnist"

    crm = CRM(experiment_dir=experiment_dir,device=torch.device("cpu"))
    generative_sample = crm.pipeline(32)
    print(generative_sample)

