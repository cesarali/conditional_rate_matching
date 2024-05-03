from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import DiscreteCIFAR10Config
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, BasicPipelineConfig, TemporalNetworkToRateConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import DiffusersUnet2DConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable


def experiment_cifar10_config(epochs=100,temporal_network_name="unet"):
    batch_size = 32
    config = CRMConfig()
    config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=batch_size, max_test_size=None)
    config.data1 = DiscreteCIFAR10Config(batch_size=batch_size)
    config.trainer = CRMTrainerConfig(
        number_of_epochs=epochs,
        learning_rate=1e-4,
        metrics=[]
    )
    config.pipeline = BasicPipelineConfig(number_of_steps=1000)
    config.temporal_network = DiffusersUnet2DConfig(num_res_blocks=2,
                                                    num_scales=4,
                                                    ch=128,
                                                    ch_mult=[1, 2, 2, 2],
                                                    input_channels=3,
                                                    scale_count_to_put_attn=1,
                                                    data_min_max=[0, 255],
                                                    dropout= 0.1,
                                                    skip_rescale=True,
                                                    time_embed_dim=128,
                                                    time_scale_factor=1000,
                                                    ema_decay=0.9999)

    config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="empty")
    return config


if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    config = experiment_cifar10_config(10,temporal_network_name="unet")

    config.trainer.debug = False
    config.trainer.device = "cuda:1"

    #config.trainer.metrics.append(MetricsAvaliable.loss_variance_times)

    call_trainer(config,
                 experiment_name="cifar_experiment",
                 experiment_type="crm_cifar10",
                 experiment_indentifier=None)
