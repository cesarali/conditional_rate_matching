from pprint import pprint
from dataclasses import asdict
from conditional_rate_matching.configs.config_crm import CRMConfig, BasicTrainerConfig, ConstantProcessConfig, BasicPipelineConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer_dario import CRMTrainer
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunityConfig, CommunitySmallConfig

# experiment_files = ExperimentFiles(experiment_name="crm",
#                                    experiment_type="graph",
#                                    experiment_indentifier="test_MLP",
#                                    delete=True)

# def experiment_graph_small_community(batch_size = 20):

#     crm_config = CRMConfig()
#     crm_config.trainer = BasicTrainerConfig(number_of_epochs = 1000,
#                                             save_model_epochs = 50,
#                                             save_metric_epochs = 50,
#                                             learning_rate = 0.0001,
#                                             device = "cuda:1",
#                                             metrics = ["mse_histograms",
#                                                        "binary_paths_histograms",
#                                                        "marginal_binary_histograms",
#                                                        "graphs_plot"])

#     crm_config.data1 = CommunitySmallConfig(dataset_name = "community_small",
#                                             batch_size=batch_size,
#                                             full_adjacency=False,
#                                             flatten=True,
#                                             as_image=False,
#                                             max_training_size=None,
#                                             max_test_size=2000)


#     crm_config.data0 = StatesDataloaderConfig(dataset_name = "categorical_dirichlet", 
#                                               dirichlet_alpha = 100.,
#                                               batch_size = batch_size,
#                                               as_image = False)
    
#     crm_config.temporal_network.hidden_dim = 256
#     crm_config.temporal_network.time_embed_dim = 32
#     crm_config.temporal_network = TemporalMLPConfig()
#     crm_config.pipeline.number_of_steps = 100
#     return crm_config

# config = experiment_graph_small_community()
# crm = CRMTrainer(config, experiment_files)
# crm.train() 


import optuna

iteration = -1

def objective(trial):

    global iteration
    iteration += 1

    experiment_files = ExperimentFiles(experiment_name="crm",
                                experiment_type="graph",
                                experiment_indentifier="optuna_scan_MLP_"+str(iteration),
                                delete=True)

    batch_size = trial.suggest_int('batch_size', 5, 50)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 512)
    time_embed_dim = trial.suggest_int('time_embed_dim', 8, 64)
    gamma = trial.suggest_float('gamma', 0.0, 2.0)

    crm_config = CRMConfig()
        
    crm_config.data1 = CommunitySmallConfig(dataset_name="community_small",
                                            batch_size=batch_size,
                                            full_adjacency=False,
                                            flatten=True,
                                            as_image=False,
                                            max_training_size=None,
                                            max_test_size=2000)
    
    crm_config.data0 = StatesDataloaderConfig(dataset_name="categorical_dirichlet",
                                              dirichlet_alpha=100.,
                                              batch_size=batch_size,
                                              as_image=False)
    
    crm_config.process = ConstantProcessConfig(gamma=gamma)
    crm_config.temporal_network = TemporalMLPConfig()

    crm_config.trainer = BasicTrainerConfig(number_of_epochs=500,
                                            save_model_epochs=50,
                                            save_metric_epochs=50,
                                            learning_rate=learning_rate,
                                            device="cuda:1",
                                            metrics=["mse_histograms", 
                                                     "binary_paths_histograms", 
                                                     "marginal_binary_histograms", 
                                                     "graphs_plot"])
    
    crm_config.temporal_network.hidden_dim = hidden_dim
    crm_config.temporal_network.time_embed_dim = time_embed_dim
    crm_config.pipeline.number_of_steps = 150

    # Train the model
    crm = CRMTrainer(crm_config, experiment_files)
    _ , metrics = crm.train()
    mse = metrics["mse_marginal_histograms"]

    return mse

# Create a study object and optimize the objective function
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Print the best parameters
print(study.best_params)