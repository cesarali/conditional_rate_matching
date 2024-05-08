import os
import torch
import pytest
from pprint import pprint

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.data.music_dataloaders import LankhPianoRollDataloader
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_music import experiment_music_conditional_config
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network

def test_conditional_pipeline():
    bridge_conditional = False
    config = experiment_music_conditional_config(bridge_conditional=bridge_conditional)
    crm = CRM(config)

    databatch = next(crm.dataloader_0.train().__iter__())
    x = databatch[0]
    batch_size = x.size(0)
    ts = torch.rand((batch_size,))
    print(x.shape)

    logits = crm.forward_rate.temporal_network(x,ts)
    print(logits.shape)

    generative_sample,original_sample = crm.pipeline(32,origin=True)
    print(generative_sample.shape)


