import torch
import pytest
from conditional_rate_matching.data.image_dataloader_config import DiscreteCIFAR10Config
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_Cifar import experiment_cifar10_config

def test_cifar():
    config = experiment_cifar10_config()
    device = config.trainer.device if torch.cuda.is_available() else torch.device("cpu")
    crm = CRM(config=config,device=device)
    databatch = next(crm.dataloader_1.train().__iter__())
    images_ = databatch[0]
    print(images_.shape)
    print(images_.max())

    time = torch.rand((images_.size(0))).to(crm.device)
    h = crm.forward_rate.temporal_network(images_.float(),time)
    print(h.shape)



