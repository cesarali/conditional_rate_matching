import torch
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph_bridge import experiment_comunity_small_bridge
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph_bridge import experiment_ego_bridge
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph_bridge import experiment_grid_bridge

from conditional_rate_matching.data.utils.bridge_data import obtain_power_law_graph
from conditional_rate_matching.data.utils.bridge_data import obtain_graph_dataset
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
import pytest

def test_graph():
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.models.generative_models.crm import CRM
    from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph_test")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    config = experiment_comunity_small_bridge(number_of_epochs=50,network="gnn")
    #config = experiment_grid_bridge(number_of_epochs=50,network="gnn")
    #config = experiment_ego_bridge(number_of_epochs=50,network="gnn")

    generative_model = CRM(config,experiment_files=experiment_files)
    databatch = next(generative_model.dataloader_1.train().__iter__())
    print(databatch[0].shape)
    databatch = next(generative_model.dataloader_0.train().__iter__())
    print(databatch[0].shape)
    #dataloader_0,dataloader_1 = get_dataloaders_crm(config)
    #train_graphs,test_graphs = obtain_graph_dataset(generative_model.dataloader_1)
    #print(len(train_graphs))
    #print(len(test_graphs))
