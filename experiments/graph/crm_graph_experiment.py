from pprint import pprint
from dataclasses import asdict
import datetime
import numpy as np
import os

import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour, plot_parallel_coordinate, plot_param_importances

from conditional_rate_matching.configs.config_crm import CRMConfig, BasicTrainerConfig, ConstantProcessConfig, BasicPipelineConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer_dario import CRMTrainer
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig, TemporalDeepMLPConfig, ConvNetAutoencoderConfig
from conditional_rate_matching.data.graph_dataloaders_config import CommunityConfig, CommunitySmallConfig


def experiment_graph_small_community(batch_size = 20):

    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph",
                                       experiment_indentifier="test_MLP",
                                       delete=True)

    crm_config = CRMConfig()
    crm_config.trainer = BasicTrainerConfig(number_of_epochs = 1000,
                                            save_model_epochs = 50,
                                            save_metric_epochs = 50,
                                            learning_rate = 0.0001,
                                            device = "cuda:1",
                                            metrics = ["mse_histograms",
                                                       "binary_paths_histograms",
                                                       "marginal_binary_histograms",
                                                       "graphs_plot"])

    crm_config.data1 = CommunitySmallConfig(dataset_name = "community_small",
                                            batch_size=batch_size,
                                            full_adjacency=False,
                                            flatten=True,
                                            as_image=False,
                                            max_training_size=None,
                                            max_test_size=2000)


    crm_config.data0 = StatesDataloaderConfig(dataset_name = "categorical_dirichlet", 
                                              dirichlet_alpha = 100.,
                                              batch_size = batch_size,
                                              as_image = False)
    
    crm_config.temporal_network.hidden_dim = 256
    crm_config.temporal_network.time_embed_dim = 32
    crm_config.temporal_network.num_layers = 2
    crm_config.temporal_network.activation = "ReLU"
    crm_config.temporal_network = TemporalDeepMLPConfig()
    crm_config.pipeline.number_of_steps = 100
    crm = CRMTrainer(crm_config, experiment_files)
    crm.train() 

class ScanOptuna:
    def __init__(self, 
                 experiment_type="graph",
                 experiment_indentifier="optuna_scan",
                 device="cpu",
                 n_trials=100,
                 epochs=500,
                 batch_size=(5, 50),
                 learning_rate=(1e-5, 1e-2), 
                 hidden_dim=(16, 512), 
                 num_layers=(1, 5),
                 activation=("ReLU", "LeakyReLU"),
                 time_embed_dim=(8, 64), 
                 gamma=(0.0, 2.0)):

        # params
        self.experiment_type = experiment_type
        self.experiment_indentifier = experiment_indentifier
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = list(activation)
        self.time_embed_dim = time_embed_dim
        self.gamma = gamma
        self.iteration = 0
        self.metric = np.inf
        # scan
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=n_trials)
        print(self.study.best_params)
        
    def name(self, var, scope=globals()):
        for name, value in scope.items():
            if value is var:
                return name
        return None

    def def_param(self, trial, name, param, type):
        if type == "int":
            return trial.suggest_int(name, param[0], param[1]) if isinstance(param, tuple) else param
        if type == "cat":
            return trial.suggest_categorical(name, param) if isinstance(param, list) else param
        if type == "float":
            if not isinstance(param, tuple): return param
            elif param[0] / param[1] > 0.05: return trial.suggest_float(name, param[0], param[1])
            else: return trial.suggest_float(name, param[0], param[1], log=True)
        
    def objective(self, trial):

        self.iteration += 1
        exp_id = self.experiment_indentifier + "_" + str(self.iteration)

        self.experiment_files = ExperimentFiles(experiment_name="crm",
                                                experiment_type=self.experiment_type,
                                                experiment_indentifier=exp_id,
                                                delete=True)

        epochs = self.def_param(trial, 'epochs', self.epochs, type="int")
        batch_size = self.def_param(trial, 'bach_size', self.batch_size, type="int")
        learning_rate = self.def_param(trial, 'lr', self.learning_rate, type="float")
        hidden_dim = self.def_param(trial, 'dim_hid', self.hidden_dim, type="int")
        num_layers = self.def_param(trial, 'num_layers', self.num_layers, type="int")
        activation = self.def_param(trial, 'activation', self.activation, type="cat")
        time_embed_dim = self.def_param(trial, 'dim_t_emb', self.time_embed_dim, type="int")
        gamma = self.def_param(trial, 'gamma', self.gamma, type="float")

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
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation)

        crm_config.trainer = BasicTrainerConfig(number_of_epochs=epochs,
                                                learning_rate=learning_rate,
                                                device=self.device,
                                                metrics=["mse_histograms", 
                                                        "binary_paths_histograms", 
                                                        "marginal_binary_histograms", 
                                                        "graphs_plot"])
        
        crm_config.pipeline.number_of_steps = 150

        # Train the model
        crm = CRMTrainer(crm_config, self.experiment_files)

        _ , metrics = crm.train()

        self.mse = metrics["mse_marginal_histograms"]
        if self.mse < self.metric: self.metric = self.mse
        else: os.system("rm -rf /home/df630/conditional_rate_matching/results/crm/{}/{}".format(self.experiment_type, exp_id))
        
        return self.mse


if __name__ == "__main__":

    scan = ScanOptuna(
                    experiment_type="graph_"+str(datetime.datetime.now().strftime("%Y.%m.%d-%Hh%Ms%S")),
                    experiment_indentifier="optuna_scan_MLP",
                    n_trials=2000,
                    epochs=2000,
                    batch_size=(8, 64),
                    learning_rate=(1e-7, 1e-3), 
                    hidden_dim=(16, 128), 
                    num_layers=(2, 6),
                    activation=('ReLU', 'LeakyReLU', 'GELU', 'SELU', 'CELU', 'ELU', 'PReLU'),
                    time_embed_dim=(16, 128), 
                    gamma=(0.00001, 1000),
                    device='cuda:1'
                    )
    
    workdir = scan.experiment_files.experiment_type_dir

    df = scan.study.trials_dataframe()
    df.to_csv(workdir + '/trials.tsv', sep='\t', index=False)

    # Save Optimization History
    fig = plot_optimization_history(scan.study)
    fig.write_image(workdir + "/optimization_history.png")

    # Save Slice Plot
    fig = plot_slice(scan.study)
    fig.write_image(workdir + "/slice_plot.png")

    # Save Contour Plot
    fig = plot_contour(scan.study)
    fig.write_image(workdir + "/contour_plot.png")

    # Save Parallel Coordinate Plot
    fig = plot_parallel_coordinate(scan.study)
    fig.write_image(workdir + "/parallel_coordinate.png")

    # Save Parameter Importances
    fig = plot_param_importances(scan.study)
    fig.write_image(workdir + "/param_importances.png")

