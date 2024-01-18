from pprint import pprint

from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable

from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

from conditional_rate_matching.configs.configs_classes.config_crm import BasicTrainerConfig

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import ConvNetAutoencoderConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import UConvNISTNetConfig

def experiment_nist(number_of_epochs=300,
                    dataset_name="emnist",
                    temporal_network_name="mlp",
                    berlin=True):

    ctdd_config = CTDDConfig()
    if temporal_network_name == "mlp":
        ctdd_config.data0 = NISTLoaderConfig(flatten=True,as_image=False,batch_size=128,dataset_name=dataset_name)
        ctdd_config.temporal_network = TemporalMLPConfig()
    elif temporal_network_name == "unet_conv":
        ctdd_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=128, dataset_name=dataset_name)
        ctdd_config.temporal_network = UConvNISTNetConfig()

    ctdd_config.pipeline.number_of_steps = 100
    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=number_of_epochs,
                                             berlin=berlin,
                                             metrics=[MetricsAvaliable.mse_histograms,
                                                      MetricsAvaliable.mnist_plot,
                                                      MetricsAvaliable.fid_nist,
                                                      MetricsAvaliable.marginal_binary_histograms],
                                             learning_rate=1e-4)
    return ctdd_config

if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    config = experiment_nist(2,"emnist",temporal_network_name="mlp")
    config.trainer.debug = True
    pprint(config)
    config.pipeline.number_of_steps = 10
    #config.trainer.device = "cuda:0"

    call_trainer(config)
