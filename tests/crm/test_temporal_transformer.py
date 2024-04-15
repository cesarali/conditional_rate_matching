import os
import torch
import pytest
from pprint import pprint

from conditional_rate_matching.data.music_dataloaders import LankhPianoRollDataloader
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_music import experiment_music_conditional_config
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network

def test_temporal_transformer():
    config = experiment_music_conditional_config()
    data_loader = LankhPianoRollDataloader(config.data1)

    databatch = next(data_loader.train().__iter__())
    print("data_0")
    print(databatch[0][0].shape)

    print("data_1")
    print(databatch[1][0].shape)

    databatch = next(data_loader.data1.train().__iter__())
    x = databatch[0]
    batch_size = x.size(0)
    ts = torch.rand((batch_size,))

    device = torch.device(config.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
    config.temporal_network.num_layers = 2
    config.temporal_network.num_heads = 1

    temporal_network = load_temporal_network(config,device)
    logits = temporal_network(x,ts)

    print(logits.shape)



