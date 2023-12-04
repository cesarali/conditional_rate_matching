import os
from pprint import pprint
from dataclasses import asdict

from conditional_rate_matching.configs.config_crm import CRMConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.config_crm import BasicTrainerConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import ConvNetAutoencoderConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import AvailableGrayCodes
AvailableGrayCodes.checkerboard
def experiment_nist(number_of_epochs=300,
                    dataset_name="checkerboard",
                    temporal_network_name="mlp",
                    berlin=True):
    crm_config = CRMConfig()
    if temporal_network_name == "mlp":
        crm_config.data1 = GrayCodesDataloaderConfig(dataset_name=dataset_name,batch_size=64)
        crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=64)
        crm_config.temporal_network = TemporalMLPConfig()
    crm_config.pipeline.number_of_steps = 5
    crm_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs,
                                            berlin=berlin,
                                            metrics=[MetricsAvaliable.kdmm,
                                                     MetricsAvaliable.marginal_binary_histograms,
                                                     MetricsAvaliable.grayscale_plot],
                                            max_test_size=4000,#size of test sample for measuring distance
                                            learning_rate=1e-4)
    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    config = experiment_nist(4,AvailableGrayCodes.swissroll)
    config.trainer.debug = True

    call_trainer(config)