import torch
import pytest

from pprint import pprint

def test_unet():
    from conditional_rate_matching.configs.config_crm import CRMConfig
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
    from conditional_rate_matching.models.generative_models.crm import CRM
    from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="trainer_call")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    config : CRMConfig
    config = experiment_nist(temporal_network_name="unet")
    pprint(config)
    #config.data1 = NISTLoaderConfig(flatten=False,as_image=True)
    #config.data1 = NISTLoaderConfig(flatten=False,as_image=True)

    generative_model = CRM(config,experiment_files=experiment_files)
    databatch = next(generative_model.dataloader_1.train().__iter__())
    x = databatch[0].to(device)
    t = torch.rand(x.size(0),).to(device)

    rate_ = generative_model.forward_rate(x,t)
    print(rate_.shape)



