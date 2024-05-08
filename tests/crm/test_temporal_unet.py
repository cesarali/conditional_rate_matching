import torch
from pprint import pprint

def test_unet():
    from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.models.generative_models.crm import CRM
    from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="trainer_call")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    config : CRMConfig
    config = experiment_nist(temporal_network_name="mlp",dataset_name="mnist")
    pprint(config)
    #config.data1 = NISTLoaderConfig(flatten=False,as_image=True)
    #config.data1 = NISTLoaderConfig(flatten=False,as_image=True)

    generative_model = CRM(config,experiment_files=experiment_files)
    databatch = next(generative_model.dataloader_1.train().__iter__())
    x = databatch[0].to(device)
    t = torch.rand(x.size(0),).to(device)

    rate_ = generative_model.forward_rate(x,t)
    print(rate_.mean())

def test_crm_unet():
    from conditional_rate_matching.models.generative_models.crm import CRM
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
    from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
    from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist
    from conditional_rate_matching.models.temporal_networks.unet import UNetModelWrapper
    from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="trainer_call")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    config : CRMConfig
    config = experiment_nist(temporal_network_name="cfm_unet",dataset_name="mnist")
    data_loader_0,data_loader_1,_ = get_dataloaders_crm(config)
    temporal_network = load_temporal_network(config,device)

    databatch = next(data_loader_1.train().__iter__())
    x_0 = databatch[0].reshape(-1,1,28,28)
    t = torch.rand((x_0.size(0),))

    output = temporal_network(t,x_0)
    print(output.shape)








