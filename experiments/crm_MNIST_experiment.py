from pprint import pprint
from dataclasses import asdict
from conditional_rate_matching.configs.config_crm import CRMConfig, BasicTrainerConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.image_dataloaders import NISTLoader
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer_dario import CRMTrainer
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import ConvNetAutoencoderConfig, TemporalMLPConfig



# Files to save the experiments
experiment_files = ExperimentFiles(experiment_name="crm",
                                   experiment_type="MNIST",
                                   experiment_indentifier="test_MLP",
                                   delete=True)

def experiment_MNIST(as_image = False,
                     batch_size = 128,
                     max_training_size=20000,
                     max_test_size=2000):
    """
    MNIST EXPERIMENT
    :return:
    """
    crm_config = CRMConfig()
    crm_config.trainer = BasicTrainerConfig(number_of_epochs = 100,
                                            save_model_epochs = 50,
                                            save_metric_epochs = 50,
                                            learning_rate = 0.001,
                                            device = "cuda:1",
                                            metrics = ["mse_histograms",
                                                       "binary_paths_histograms",
                                                       "marginal_binary_histograms",
                                                       "mnist_plot"])

    crm_config.data1 = NISTLoaderConfig(dataset_name = "mnist",
                                        batch_size = batch_size,
                                        vocab_size = 2,
                                        flatten = False if as_image else True,
                                        as_image = as_image,
                                        max_training_size = max_training_size,
                                        max_test_size = max_test_size)
    
    crm_config.data0 = StatesDataloaderConfig(dataset_name = "categorical_dirichlet", 
                                              dimensions = int(28*28),
                                              vocab_size = 2,
                                              dirichlet_alpha = 100.,
                                              batch_size = batch_size,
                                              as_image = as_image,
                                              sample_size = max_training_size,
                                              max_test_size = max_test_size)
    
    crm_config.temporal_network.hidden_dim = 32
    crm_config.temporal_network.time_embed_dim = 16
    crm_config.temporal_network = ConvNetAutoencoderConfig() if as_image else TemporalMLPConfig()

    return crm_config

config = experiment_MNIST()
crm = CRMTrainer(config, experiment_files)
crm.train() 