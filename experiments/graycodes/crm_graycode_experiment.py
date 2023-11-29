from multiprocessing import pool
from pprint import pprint
from dataclasses import asdict
import datetime
from matplotlib.pyplot import gray
import numpy as np
import os
import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour, plot_parallel_coordinate, plot_param_importances

from conditional_rate_matching.configs.config_crm import CRMConfig, BasicTrainerConfig, ConstantProcessConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig
from conditional_rate_matching.data.gray_codes_dataloaders_config import AvailableGrayCodes


def CRM_single_run(dynamics="crm",
                    experiment_type="graycode",
                    experiment_indentifier="run",
                    model="mlp",
                    dataset0=None,
                    dataset1=AvailableGrayCodes.swissroll,
                    metrics=[MetricsAvaliable.marginal_binary_histograms,
                             MetricsAvaliable.grayscale_plot],
                    device="cpu",
                    epochs=100,
                    batch_size=256,
                    learning_rate=1e-3, 
                    hidden_dim=64, 
                    num_layers=2,
                    activation="ReLU",
                    time_embed_dim=8,
                    num_timesteps=64):

    databridge = dataset0 + "_" + dataset1 if dataset0 is not None else dataset1
    experiment_type = experiment_type + "_" + databridge + "_" + model + "_" + str(datetime.datetime.now().strftime("%Y.%m.%d_%Hh%Ms%S"))
    experiment_files = ExperimentFiles(experiment_name=dynamics,
                                       experiment_type=experiment_type,
                                       experiment_indentifier=experiment_indentifier,
                                       delete=True)
    #...configs:

    crm_config = CRMConfig()

    if dataset0 is None: crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=batch_size)
    else: crm_config.data0 = GrayCodesDataloaderConfig(dataset_name=dataset0, batch_size=batch_size)
    
    crm_config.data1 = GrayCodesDataloaderConfig(dataset_name=dataset1, batch_size=batch_size)
    
    if model=="mlp":
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation)
        
    if model=="deepEBM":
        pass
    
    crm_config.trainer = BasicTrainerConfig(number_of_epochs=epochs,
                                            learning_rate=learning_rate,
                                            device=device,
                                            metrics=metrics)
    
    crm_config.pipeline.number_of_steps = num_timesteps

    #...train

    crm = CRMTrainer(crm_config, experiment_files)
    _ , metrics = crm.train()
    return metrics


class CRM_Scan_Optuna:
    def __init__(self, 
                 dynamics="crm",
                 experiment_type="graycode",
                 experiment_indentifier="optuna_scan",
                 model="mlp",
                 device="cpu",
                 n_trials=100,
                 epochs=500,
                 batch_size=(5, 50),
                 learning_rate=(1e-5, 1e-2), 
                 hidden_dim=(16, 512), 
                 num_layers=(1, 5),
                 activation=("ReLU", "LeakyReLU"),
                 time_embed_dim=(8, 64), 
                 gamma=(0.0, 2.0),
                 num_timesteps=100):

        # params
        self.dynamics = dynamics
        self.experiment_type = experiment_type + "_" + model + "_" + str(datetime.datetime.now().strftime("%Y.%m.%d_%Hh%Ms%S"))
        self.experiment_indentifier = experiment_indentifier
        self.workdir = "/home/df630/conditional_rate_matching/results/{}/{}".format(dynamics, self.experiment_type)
        self.model = model
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = list(activation)
        self.time_embed_dim = time_embed_dim
        self.gamma = gamma
        self.num_timesteps = num_timesteps

        self.iteration, self.metric = 0, np.inf
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

        self.experiment_files = ExperimentFiles(experiment_name=self.dynamics,
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
        crm_config.data0 = GrayCodesDataloaderConfig(dataset_name="gaussians",batch_size=batch_size)
        
        crm_config.data1 = GrayCodesDataloaderConfig(dataset_name="swissroll",batch_size=batch_size)
        
        crm_config.process = ConstantProcessConfig(gamma=gamma)

        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation)

        crm_config.trainer = BasicTrainerConfig(number_of_epochs=epochs,
                                                learning_rate=learning_rate,
                                                device=self.device,
                                                metrics=["grayscale_plot"])
        
        crm_config.pipeline.number_of_steps = self.num_timesteps

        # Train the model
        crm = CRMTrainer(crm_config, self.experiment_files)
        _ , metrics = crm.train()
        print('all metric: ', metrics)
        if self.graph_metric < self.metric: self.metric = self.graph_metric
        else: os.system("rm -rf {}/{}".format(self.workdir, exp_id))
        
        return self.graph_metric


if __name__ == "__main__":


    '''
    swissroll, circles, moons, gaussians, pinwheel, spirals, checkerboard, line, cos

    '''


    CRM_single_run(dataset0=AvailableGrayCodes.swissroll, 
                   dataset1=AvailableGrayCodes.checkerboard,
                   metrics=[MetricsAvaliable.mse_histograms,
                             MetricsAvaliable.marginal_binary_histograms,
                             MetricsAvaliable.kdmm,
                             MetricsAvaliable.grayscale_plot],
                   epochs=100,
                   batch_size=20,
                   learning_rate=1e-3, 
                   hidden_dim=64, 
                   num_layers=2,
                   activation="ReLU",
                   time_embed_dim=8,
                   device="cuda:0")
