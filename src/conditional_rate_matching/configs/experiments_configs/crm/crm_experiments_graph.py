from conditional_rate_matching.configs.configs_classes.config_crm import OptimalTransportSamplerConfig

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, TemporalNetworkToRateConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunitySmallConfig, EgoConfig, GridConfig, EnzymesConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import TemporalScoreNetworkAConfig,SimpleTemporalGCNConfig

"""

    The following functions create config files for experiments with graph data

"""

def experiment_ego(number_of_epochs=300, berlin=True, network="mlp",temporal_to_rate=None):
    crm_config = CRMConfig()
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.0, batch_size=20)
    crm_config.pipeline.number_of_steps = 1000

    crm_config.trainer = CRMTrainerConfig(
        number_of_epochs=number_of_epochs,
        windows=berlin,
        metrics=[MetricsAvaliable.mse_histograms, MetricsAvaliable.graphs_plot, MetricsAvaliable.marginal_binary_histograms],
        learning_rate=1e-2,
    )

    if network == "gnn":
        crm_config.data1 = EgoConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.trainer.learning_rate = 1e-2
    elif network == "simple":
        crm_config.data1 = EgoConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = SimpleTemporalGCNConfig()
        crm_config.trainer.learning_rate = 1e-3
    else:
        crm_config.data1 = EgoConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 50
        crm_config.temporal_network.time_embed_dim = 50
        crm_config.trainer.learning_rate = 1e-4

    if temporal_to_rate == "linear":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="linear",linear_reduction=.5)
    elif temporal_to_rate == "bernoulli":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="linear")
    
    return crm_config


def experiment_comunity_small(number_of_epochs=300, berlin=True, network="mlp", temporal_to_rate=None):
    crm_config = CRMConfig()
    crm_config.pipeline.number_of_steps = 1000

    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.0, batch_size=20)
    crm_config.trainer = CRMTrainerConfig(
        number_of_epochs=number_of_epochs,
        windows=berlin,
        metrics=[MetricsAvaliable.mse_histograms, MetricsAvaliable.graphs_plot, MetricsAvaliable.marginal_binary_histograms],
        learning_rate=1e-2,
    )

    if network == "gnn":
        crm_config.data1 = CommunitySmallConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.trainer.learning_rate = 1e-2
    if network == "simple":
        crm_config.data1 = CommunitySmallConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = SimpleTemporalGCNConfig()
        crm_config.trainer.learning_rate = 1e-3
    else:
        crm_config.data1 = CommunitySmallConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 200
        crm_config.temporal_network.time_embed_dim = 200
        crm_config.trainer.learning_rate = 1e-4

    if temporal_to_rate == "linear":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="linear",linear_reduction=.5)
    elif temporal_to_rate == "bernoulli":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="bernoulli")
    else:
        crm_config.temporal_network_to_rate = None
    return crm_config


def experiment_grid(number_of_epochs=300, berlin=True, network="mlp", temporal_to_rate=None):
    crm_config = CRMConfig()
    crm_config.temporal_network_to_rate = 0.1
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)

    crm_config.trainer = CRMTrainerConfig(
        number_of_epochs=number_of_epochs,
        windows=berlin,
        metrics=[MetricsAvaliable.mse_histograms, MetricsAvaliable.graphs_plot, MetricsAvaliable.marginal_binary_histograms],
        learning_rate=1e-2,
    )
    if network == "gnn":
        crm_config.data1 = GridConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.pipeline.number_of_steps = 1000
        crm_config.trainer.learning_rate = 1e-2
    elif network == "simple":
        crm_config.data1 = GridConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = SimpleTemporalGCNConfig()
        crm_config.trainer.learning_rate = 1e-3
    else:
        crm_config.data1 = GridConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 100
        crm_config.temporal_network.time_embed_dim = 100
        crm_config.trainer.learning_rate = 1e-4

    if temporal_to_rate == "linear":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="linear",linear_reduction=.5)
    elif temporal_to_rate == "bernoulli":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="bernoulli")
    else:
        crm_config.temporal_network_to_rate = None
    
    return crm_config

def experiment_enzymes(number_of_epochs=300, berlin=True, network="mlp", temporal_to_rate=None):
    crm_config = CRMConfig()
    crm_config.temporal_network_to_rate = 0.1
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100.,batch_size=20)

    crm_config.trainer = CRMTrainerConfig(
        number_of_epochs=number_of_epochs,
        windows=berlin,
        metrics=[MetricsAvaliable.mse_histograms, MetricsAvaliable.graphs_plot, MetricsAvaliable.marginal_binary_histograms],
        learning_rate=1e-2,
    )

    if network == "gnn":
        crm_config.data1 = EnzymesConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = TemporalScoreNetworkAConfig()
        crm_config.trainer.learning_rate = 1e-2
    elif network == "simple":
        crm_config.data1 = EnzymesConfig(flatten=False, as_image=False, full_adjacency=True, batch_size=20)
        crm_config.temporal_network = SimpleTemporalGCNConfig()
        crm_config.trainer.learning_rate = 1e-3
    else:
        crm_config.data1 = EnzymesConfig(flatten=True, as_image=False, full_adjacency=False, batch_size=20)
        crm_config.temporal_network.hidden_dim = 200
        crm_config.temporal_network.time_embed_dim = 200
        crm_config.trainer.learning_rate = 1e-3

    crm_config.pipeline.number_of_steps = 1000

    if temporal_to_rate == "linear":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="linear",linear_reduction=.5)
    elif temporal_to_rate == "bernoulli":
        crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of="bernoulli")
    else:
        crm_config.temporal_network_to_rate = None

    return crm_config

if __name__ == "__main__":
    from conditional_rate_matching.models.trainers.call_all_trainers import call_trainer

    config = experiment_comunity_small(number_of_epochs=500, network="mlp",temporal_to_rate="linear")
    # config = experiment_grid(number_of_epochs=10)
    # config = experiment_ego(number_of_epochs=10,network="gnn")
    # config = experiment_enzymes(number_of_epochs=100,network="simple")

    #config.trainer.orca_dir = "C:/Users/cesar/Desktop/Projects/DiffusiveGenerativeModelling/Codes/conditional_rate_matching/src/conditional_rate_matching/models/metrics/orca_berlin_2/"
    #config.trainer.orca_dir = "/home/cvejoski/Projects/conditional_rate_matching/src/conditional_rate_matching/models/metrics/orca"
    #config.trainer.windows = True

    config.trainer.save_model_test_stopping = True
    #config.trainer.metrics.append(MetricsAvaliable.graphs_metrics)
    #config.trainer.debug = True
    #config.temporal_network = TemporalScoreNetworkAConfig(num_layers=2,num_heads=4,depth=3)
    config.optimal_transport = OptimalTransportSamplerConfig(name="OTPlanSampler",cost="log")

    config.pipeline.set_diagonal = False
    config.thermostat.gamma = 1.
    config.trainer.learning_rate = 1e-3
    config.pipeline.number_of_steps = 100
    config.trainer.loss_regularize_variance = False
    config.trainer.device = "cpu"

    results, metrics = call_trainer(config,
                                    experiment_name="prenzlauer_experiment",
                                    experiment_type="crm_graphs",
                                    experiment_indentifier=None)
    

