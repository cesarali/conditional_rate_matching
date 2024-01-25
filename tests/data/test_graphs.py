import os

from conditional_rate_matching.data.graph_dataloaders_config import CommunityConfig,CommunitySmallConfig
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
import torch
from torch import nn
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph import experiment_comunity_small
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph import experiment_ego
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph import experiment_grid
from conditional_rate_matching.data.utils.bridge_data import obtain_power_law_graph
from conditional_rate_matching.data.utils.bridge_data import obtain_graph_dataset
import pytest

from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network
def test_graph():
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.models.generative_models.crm import CRM
    from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph_test")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    #config = experiment_comunity_small(number_of_epochs=50,network="gnn")
    config = experiment_grid(number_of_epochs=50,network="gnn")
    #config = experiment_ego(number_of_epochs=50,network="gnn")

    dataloader_0,dataloader_1 = get_dataloaders_crm(config)
    databatch = next(dataloader_1.train().__iter__())
    x_adj = databatch[0]
    time = torch.rand(x_adj.size(0))

    """
    out = nn.Linear(361 * 361, 500)
    nn.Linear(500, 361)
    temporal_network = load_temporal_network(config,torch.device("cpu"))
    h = temporal_network(x_adj,time)
    print(out(h.view(-1,361*361))
    """