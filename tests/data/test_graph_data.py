from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.data.graph_dataloaders_config import CommunityConfig,EnzymesConfig

if __name__=="__main__":
    config = CRMConfig()
    config.data1 = EnzymesConfig()
    config.data0 = EnzymesConfig()
    
    dataloader_0, dataloader_1, parent_dataloader = get_dataloaders_crm(config)