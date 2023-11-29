import os
import torch
import pytest


def test_graph():
    from conditional_rate_matching.configs.config_crm import CRMConfig
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.data.graph_dataloaders_config import EgoConfig
    from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
    from conditional_rate_matching.models.generative_models.crm import CRM
    from conditional_rate_matching.models.temporal_networks.temporal_networks_config import ConvNetAutoencoderConfig
    from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalGraphConvNetConfig

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="trainer_call")
    device = torch.device("cuda:0")
    config = CRMConfig()

    config.data1 = EgoConfig(flatten=False,full_adjacency=True,as_image=True)
    #config.data1 = NISTLoaderConfig(flatten=False,as_image=True)

    config.temporal_network = ConvNetAutoencoderConfig()
    config.temporal_network = TemporalGraphConvNetConfig()

    generative_model = CRM(config,experiment_files=experiment_files)
    databatch = next(generative_model.dataloader_1.train().__iter__())
    x = databatch[0].to(device)
    t = torch.rand(x.size(0),).to(device)
    print(x.shape)
    #fr = generative_model.forward_rate(x,t)
    #print(fr.shape)
