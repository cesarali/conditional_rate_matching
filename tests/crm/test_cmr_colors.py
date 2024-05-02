import os
from pprint import pprint
from dataclasses import dataclass, asdict
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
from conditional_rate_matching.models.generative_models.crm import CRM

def get_basic_colors_config():
    config = CRMConfig()
    config.data0 = StatesDataloaderConfig(dimensions=3,vocab_size=3)
    config.data1 = StatesDataloaderConfig(dirichlet_alpha=0.001,total_data_size=1000,dimensions=3,vocab_size=3,test_size=0.1)
    return config

if __name__=="__main__":
    config = get_basic_colors_config()
    dataloader_0,dataloader_1,_ = get_dataloaders_crm(config)
    crm = CRM(config=config)

    databatch = next(dataloader_0.train().__iter__())
    print(databatch[0])