import os
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.gray_codes_dataloaders_config import AvailableGrayCodes
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,CRMTrainerConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig

def experiment_graycode(number_of_epochs=300,
                    dataset_name="checkerboard",
                    temporal_network_name="mlp",
                    berlin=True):
    crm_config = CRMConfig()
    if temporal_network_name == "mlp":
        crm_config.data1 = GrayCodesDataloaderConfig(dataset_name=dataset_name,batch_size=128,training_size=2000,test_size=500)
        crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=128)
        crm_config.temporal_network = TemporalMLPConfig(hidden_dim=50,time_embed_dim=50)
    crm_config.pipeline.number_of_steps = 100
    crm_config.optimal_transport.name = "uniform"
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          windows=berlin,
                                          metrics=[MetricsAvaliable.kdmm,
                                                     MetricsAvaliable.marginal_binary_histograms,
                                                     MetricsAvaliable.grayscale_plot],
                                          save_model_metrics_stopping=True,
                                          metric_to_save="kdmm",
                                          max_test_size=4000,  #size of test sample for measuring distance
                                          learning_rate=1e-4)
    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    config = experiment_graycode(5,AvailableGrayCodes.checkerboard)
    config.trainer.debug = True

    call_trainer(config,
                 experiment_name="prenzlauer_experiment",
                 experiment_type="crm",
                 experiment_indentifier=None)
