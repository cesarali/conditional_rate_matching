from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, BasicPipelineConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import SequenceTransformerConfig
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.models.networks.mlp_config import MLPConfig
from conditional_rate_matching.configs.configs_classes.config_crm import OptimalTransportSamplerConfig

def experiment_music_config(epochs=100, temporal_network_name="unet"):
    batch_size = 128
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

def experiment_music_conditional_config(epochs=10,number_of_steps=10, temporal_network_name="transformer", bridge_conditional=False):
    batch_size = 64
    config = CRMConfig()
    config.data0 = LakhPianoRollConfig(batch_size=batch_size,
                                       conditional_model=True,
                                       bridge_conditional=bridge_conditional)
    config.data0.max_test_size = 950

    config.data1 = config.data0

    if temporal_network_name == "mlp":
        config.temporal_network = TemporalDeepMLPConfig(time_embed_dim=150,
                                                        hidden_dim=200)
    if temporal_network_name == "transformer":
        config.temporal_network = SequenceTransformerConfig()


    config.trainer = CRMTrainerConfig(
                                    number_of_epochs=epochs,
                                    learning_rate=0.0002,
                                    metrics=[ MetricsAvaliable.music_plot])
    config.pipeline = BasicPipelineConfig(number_of_steps=5)
    config.optimal_transport = OptimalTransportSamplerConfig(name="uniform", method='exact')
    return config


if __name__=="__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer
    from conditional_rate_matching.models.temporal_networks.temporal_networks_config import SequenceTransformerConfig

    config = experiment_music_conditional_config(2000, number_of_steps=1000, temporal_network_name="transformer")

    config.temporal_network = SequenceTransformerConfig(num_layers=6,num_heads=8)
    config.trainer.debug = False
    config.trainer.device = "cuda:3"

    call_trainer(config,
                 experiment_name="test_piano_roll_transformer_2000_Epochs",
                 experiment_type="crm",
                 experiment_indentifier=None)
