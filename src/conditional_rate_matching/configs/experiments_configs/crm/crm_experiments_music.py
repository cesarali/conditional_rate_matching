
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, BasicPipelineConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import DiffusersUnet2DConfig
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig

def experiment_music_config(epochs=100, temporal_network_name="unet"):
    batch_size = 32
    config = CRMConfig()
    config.data0 = LakhPianoRollConfig(dirichlet_alpha=100., batch_size=batch_size)
    config.data1 = config.data0

    config.trainer = CRMTrainerConfig(
        number_of_epochs=epochs,
        learning_rate=1e-4,
        metrics=[]
    )
    config.pipeline = BasicPipelineConfig(number_of_steps=1000)
    config.temporal_network = TemporalDeepMLPConfig()

    return config

def experiment_music_conditional_config(epochs=100,temporal_network_name="transformer"):
    batch_size = 32
    config = CRMConfig()
    config.data0 = LakhPianoRollConfig(batch_size=batch_size,
                                       conditional_model=True,
                                       bridge_conditional=True)
    config.data1 = config.data0

    config.trainer = CRMTrainerConfig(
        number_of_epochs=epochs,
        learning_rate=1e-4,
        metrics=[]
    )

    config.pipeline = BasicPipelineConfig(number_of_steps=5)
    config.temporal_network = TemporalDeepMLPConfig()
    return config




if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer

    config = experiment_music_conditional_config(10,temporal_network_name="mlp")

    config.trainer.debug = True
    config.trainer.device = "cpu"

    #config.trainer.metrics.append(MetricsAvaliable.loss_variance_times)

    call_trainer(config,
                 experiment_name="prenzlauer_experiment",
                 experiment_type="crm_music",
                 experiment_indentifier=None)
