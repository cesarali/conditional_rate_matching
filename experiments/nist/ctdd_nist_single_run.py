from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.ctdd_trainer import CTDDTrainer
from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig, BasicTrainerConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig, TemporalLeNet5Config, TemporalLeNet5AutoencoderConfig, TemporalUNetConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

def CTDD_single_run(dynamics="ctdd",
                    experiment_type="nist",
                    experiment_indentifier="run",
                    model="mlp",
                    dataset0="mnist",
                    metrics=[MetricsAvaliable.mse_histograms, 
                             MetricsAvaliable.mnist_plot, 
                             MetricsAvaliable.marginal_binary_histograms],
                    device="cpu",
                    epochs=100,
                    batch_size=64,
                    learning_rate=1e-3, 
                    hidden_dim=256,
                    time_embed_dim=128,
                    dropout=0.1,
                    num_layers=3,
                    activation="ReLU",
                    gamma=1.0,
                    num_timesteps=1000,
                   ):

    experiment_files = ExperimentFiles(experiment_name=dynamics,
                                       experiment_type=experiment_type,
                                       experiment_indentifier=experiment_indentifier,
                                       delete=True)
    #...configs:

    ctdd_config = CTDDConfig()

    if model=="mlp":
        ctdd_config.data0 = NISTLoaderConfig(flatten=True, as_image=False, batch_size=batch_size, dataset_name=dataset0)
        ctdd_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation,
                                                            dropout = dropout)
    
    if model=="lenet5":
        ctdd_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        ctdd_config.temporal_network = TemporalLeNet5Config(hidden_dim = hidden_dim, time_embed_dim = time_embed_dim)

    if model=="lenet5Autoencoder":
        ctdd_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        ctdd_config.temporal_network = TemporalLeNet5AutoencoderConfig(hidden_dim = hidden_dim, time_embed_dim = time_embed_dim)

    if model=="unet":
        ctdd_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        ctdd_config.temporal_network = TemporalUNetConfig(hidden_dim = hidden_dim, time_embed_dim = time_embed_dim)

    ctdd_config.trainer = BasicTrainerConfig(number_of_epochs=epochs,
                                             device=device,
                                             metrics=metrics,
                                             learning_rate=learning_rate)
    
    ctdd_config.pipeline.number_of_steps = num_timesteps

    #...train

    ctdd = CTDDTrainer(ctdd_config, experiment_files)
    _ , metrics = ctdd.train()

    print('metrics=',metrics)
    return metrics

if __name__ == "__main__":

    # CTDD_single_run(dynamics="ctdd",
    #                experiment_type="mnist",
    #                model="mlp",
    #                epochs=15,
    #                dataset0="mnist",
    #                metrics=["mse_histograms", 
    #                         "fid_nist", 
    #                         "mnist_plot", 
    #                         "marginal_binary_histograms"],
    #                batch_size=256,
    #                learning_rate=0.001,
    #                hidden_dim=128,
    #                time_embed_dim=128,
    #                activation="ReLU", 
    #                num_layers=6,
    #                dropout=0.05,
    #                num_timesteps=1000,
    #                device="cuda:1")

    CTDD_single_run(dynamics="ctdd",
                   experiment_type="mnist_unet",
                   model="mlp",
                   epochs=10,
                   dataset0="mnist",
                   metrics=["mse_histograms", 
                            "fid_nist", 
                            "mnist_plot", 
                            "marginal_binary_histograms"],
                   batch_size=256,
                   learning_rate=0.0001,
                   hidden_dim=256,
                   time_embed_dim=128,
                   num_timesteps=1000,
                   device="cuda:2")