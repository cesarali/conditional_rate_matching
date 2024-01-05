import os
from pprint import pprint
from dataclasses import asdict

from conditional_rate_matching.configs.config_crm import CRMConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.config_crm import CRMTrainerConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import ConvNetAutoencoderConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig

def experiment_nist(number_of_epochs=300,
                    dataset_name="emnist",
                    temporal_network_name="mlp",
                    berlin=True):
    crm_config = CRMConfig()
    if temporal_network_name == "mlp":
        crm_config.data1 = NISTLoaderConfig(flatten=True,as_image=False,batch_size=128,dataset_name=dataset_name,max_test_size=None)
        crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=128,max_test_size=None)
        crm_config.temporal_network = TemporalMLPConfig()
    elif temporal_network_name == "conv0":
        crm_config.data1 = NISTLoaderConfig(flatten=False,as_image=True, batch_size=128,dataset_name=dataset_name)
        crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=128)
        crm_config.temporal_network = ConvNetAutoencoderConfig()

    crm_config.pipeline.number_of_steps = 100
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          berlin=berlin,
                                          metrics=[MetricsAvaliable.mse_histograms,
                                                   MetricsAvaliable.mnist_plot,
                                                   MetricsAvaliable.fid_nist,
                                                   MetricsAvaliable.marginal_binary_histograms],
                                          max_test_size=200,
                                          learning_rate=1e-4)
    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    config = experiment_nist(10,"mnist")
    config.trainer.debug = False
    config.trainer.device = "cpu"
    pprint(config)
    call_trainer(config,experiment_name="nist_fid")
