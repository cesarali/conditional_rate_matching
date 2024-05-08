from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunitySmallConfig, EgoConfig, GridConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import TemporalScoreNetworkAConfig

"""
The following functions create config files for experiments with graph data

"""


def experiment_Kvariates(number_of_epochs=300, berlin=True, network="mlp"):
    crm_config = CRMConfig()
    crm_config.data0 = StatesDataloaderConfig(dimensions=10,vocab_size=5,dirichlet_alpha=100.0, batch_size=128,sample_size=5000)
    crm_config.data1 = StatesDataloaderConfig(dimensions=10,vocab_size=5,dirichlet_alpha=0.01,batch_size=128,sample_size=5000)

    crm_config.pipeline.number_of_steps = 1000
    crm_config.trainer = CRMTrainerConfig(
        number_of_epochs=number_of_epochs,
        windows=berlin,
        metrics=[MetricsAvaliable.categorical_histograms],
        learning_rate=1e-3,
    )
    if network == "deep":
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.trainer.learning_rate = 1e-4
    else:
        crm_config.temporal_network.hidden_dim = 200
        crm_config.temporal_network.time_embed_dim = 200
        crm_config.trainer.learning_rate = 1e-3

    return crm_config


if __name__ == "__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer

    config = experiment_Kvariates(number_of_epochs=200, network="mlp")

    config.trainer.save_model_test_stopping = True
    #config.trainer.metrics.append(MetricsAvaliable.graphs_metrics)

    config.temporal_network.hidden_dim = 250
    config.temporal_network.time_embed_dim = 250

    config.thermostat.gamma = 0.01
    config.trainer.learning_rate = 1e-3
    config.pipeline.number_of_steps = 1000
    config.trainer.loss_regularize_variance = False
    config.trainer.device = "cpu"

    results, metrics = call_trainer(config,
                                    experiment_name="prenzlauer_experiment",
                                    experiment_type="crm",
                                    experiment_indentifier="colors_gamma_001")
