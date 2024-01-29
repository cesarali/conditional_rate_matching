import os
import pytest
from pprint import pprint
from conditional_rate_matching.data.music_dataloaders import LankhPianoRollDataloader
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig


def test_lankh_piano():
    config = LakhPianoRollConfig(conditional_model=True)
    data_loader = LankhPianoRollDataloader(config)

    databatch = next(data_loader.train().__iter__())
    print("data_0")
    print(databatch[0][0].shape)

    print("data_1")
    print(databatch[1][0].shape)

    databatch = next(data_loader.data1.train().__iter__())
    print(databatch[0].shape)



