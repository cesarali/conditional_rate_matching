import os
import pytest
from pprint import pprint

from conditional_rate_matching.utils.plots.images_plots import mnist_grid
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist_conditional import experiment_nist_conditional
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.data.image_dataloaders_conditional_config import DistortedNISTLoaderConfig

def test_conditional_mnist():
    config = experiment_nist_conditional()
    config.data0 = DistortedNISTLoaderConfig(distortion="swirl",distortion_level=3)

    dataloader_0, dataloader_1, parent_dataloader = get_dataloaders_crm(config)

    databatch_0, databatch_1 = next(parent_dataloader.train().__iter__())

    mnist_grid(databatch_0[0])
    mnist_grid(databatch_1[0])

    print(databatch_0[0].shape)
    print(databatch_1[0].shape)




