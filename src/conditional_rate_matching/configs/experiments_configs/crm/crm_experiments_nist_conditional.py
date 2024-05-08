from pprint import pprint

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMTrainerConfig

from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.image_dataloaders_conditional_config import DistortedNISTLoaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import UConvNISTNetConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import DiffusersUnet2DConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ExponentialThermostatConfig
from conditional_rate_matching.data.image_dataloaders_conditional_config import DistortedNISTLoaderConfig


def experiment_nist_conditional(number_of_epochs=300,
                                dataset_name="mnist",
                                temporal_network_name="mlp",
                                distortion="swirl",
                                distortion_level=3,
                                berlin=True):
    crm_config = CRMConfig()
    if temporal_network_name == "mlp":
        crm_config.data1 = DistortedNISTLoaderConfig(flatten=True,
                                                     as_image=False,
                                                     batch_size=128,
                                                     dataset_name=dataset_name,
                                                     distortion=distortion,
                                                     distortion_level=distortion_level,
                                                     max_test_size=None)
        crm_config.data0 = crm_config.data1
        crm_config.temporal_network = TemporalMLPConfig()

    crm_config.pipeline.number_of_steps = 100
    crm_config.trainer = CRMTrainerConfig(number_of_epochs=number_of_epochs,
                                          windows=berlin,
                                          metrics=[MetricsAvaliable.mse_histograms,
                                                   MetricsAvaliable.mnist_plot,
                                                   MetricsAvaliable.fid_nist,
                                                   MetricsAvaliable.marginal_binary_histograms],
                                          max_test_size=200,
                                          learning_rate=1e-4)


    return crm_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    config = experiment_nist_conditional(number_of_epochs=10,
                                         dataset_name="mnist",
                                         temporal_network_name="mlp",
                                         distortion="swirl",
                                         distortion_level=3,
                                         berlin=True)
    config.temporal_network = TemporalMLPConfig(time_embed_dim=350,
                                                hidden_dim=350)
    config.trainer.debug = True
    config.trainer.device = "cpu"
    #config.trainer.metrics.append(MetricsAvaliable.loss_variance_times)

    pprint(config)
    call_trainer(config,
                 experiment_name="pren_experiment",
                 experiment_type="crm_twisted_bridge",
                 experiment_indentifier=None)
