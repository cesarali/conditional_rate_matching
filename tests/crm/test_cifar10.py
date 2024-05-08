import torch

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_Cifar import experiment_cifar10_config
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,TemporalNetworkToRateConfig

if __name__=="__main__":
    config = experiment_cifar10_config(epochs=100)
    config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="logistic")

    crm = CRM(config)
    databatch = next(crm.dataloader_1.train().__iter__())

    batch_size = databatch[0].shape[0]
    x_1 = databatch[0]
    time = torch.rand((batch_size,))
    temporal_rates = crm.forward_rate.temporal_network(x_1.float(),time)
    logits = crm.forward_rate(x_1.float(),time)

    print(logits.shape)