import torch
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph import experiment_comunity_small

def test_graph():
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.models.generative_models.crm import CRM
    from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph_test")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    config = experiment_comunity_small(number_of_epochs=50,network="gnn")

    #config.temporal_network = ConvNetAutoencoderConfig()
    #config.temporal_network = TemporalGraphConvNetConfig()
    config.temporal_network = TemporalScoreNetworkAConfig()

    generative_model = CRM(config,experiment_files=experiment_files)
    databatch = next(generative_model.dataloader_1.train().__iter__())

    x = databatch[0].to(device)
    x_1 = databatch[1].to(device)
    time = torch.rand(x.size(0),).to(device)

    temp_output = generative_model.forward_rate.temporal_network(x,time)

    print(temp_output.shape)

