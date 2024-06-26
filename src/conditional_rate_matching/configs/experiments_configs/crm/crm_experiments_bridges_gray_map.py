from pprint import pprint

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,CRMTrainerConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import AvailableGrayCodes

def experiment_nist(number_of_epochs=300,
                    dataset_name0=AvailableGrayCodes.checkerboard,
                    dataset_name1=AvailableGrayCodes.swissroll,
                    temporal_network_name="mlp",
                    berlin=True):

    crm_config = CRMConfig()
    if temporal_network_name == "mlp":
        crm_config.data1 = GrayCodesDataloaderConfig(dataset_name=dataset_name1,batch_size=128,training_size=2000,test_size=500)
        crm_config.data0 = GrayCodesDataloaderConfig(dataset_name=dataset_name0,batch_size=128,training_size=2000,test_size=500)
        #crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=128)
        crm_config.temporal_network = TemporalMLPConfig(hidden_dim=50, time_embed_dim=50)

    crm_config.pipeline.number_of_steps = 100
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          windows=berlin,
                                          metrics=[MetricsAvaliable.mse_histograms,
                                                   MetricsAvaliable.marginal_binary_histograms,
                                                   MetricsAvaliable.grayscale_plot],
                                          max_test_size=4000,
                                          learning_rate=1e-4)
    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    config = experiment_nist(50,
                             dataset_name0=AvailableGrayCodes.swissroll,
                             dataset_name1=AvailableGrayCodes.checkerboard)
    config.trainer.debug = False
    pprint(config)
    call_trainer(config)