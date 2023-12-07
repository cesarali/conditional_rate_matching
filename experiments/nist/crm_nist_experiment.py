from multiprocessing import pool
from pprint import pprint
from dataclasses import asdict
import datetime
from matplotlib.pyplot import gray
import numpy as np
import os
import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour, plot_parallel_coordinate, plot_param_importances

from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.config_crm import CRMConfig, CRMTrainerConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig, LogThermostatConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig, ConvNetAutoencoderConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig


# from conditional_rate_matching.configs.config_crm import CRMConfig, BasicTrainerConfig, ConstantProcessConfig
# from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
# from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
# from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
# from conditional_rate_matching.configs.config_files import ExperimentFiles
# from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
# from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalDeepMLPConfig, ConvNetAutoencoderConfig 


def CRM_single_run(dynamics="crm",
                    experiment_type="nist",
                    experiment_indentifier="run",
                    thermostat=None,
                    coupling_method = 'uniform',
                    model="convnet",
                    dataset1="mnist",
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

    crm_config = CRMConfig()
    crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=batch_size)

    if model=="mlp":
        crm_config.data1 = NISTLoaderConfig(flatten=True, as_image=False, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation,
                                                            dropout = dropout)
        
    if model=="convnet":
        crm_config.data1 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = ConvNetAutoencoderConfig()
    
    if thermostat == "log":  crm_config.thermostat = LogThermostatConfig()
    else: crm_config.thermostat = ConstantThermostatConfig(gamma=gamma)

    crm_config.trainer = CRMTrainerConfig(number_of_epochs=epochs,
                                          learning_rate=learning_rate,
                                          device=device,
                                          metrics=metrics,
                                          loss_regularize_square=False,
                                          loss_regularize=False)
    
    crm_config.pipeline.number_of_steps = num_timesteps
    crm_config.optimal_transport.name = coupling_method
    #...train

    crm = CRMTrainer(crm_config, experiment_files)
    _ , metrics = crm.train()
    return metrics


class CRM_Scan_Optuna:
    def __init__(self, 
                 dynamics="crm",
                 experiment_type="nist",
                 experiment_indentifier="optuna_scan",
                 dataset1='mnist',
                 thermostat=None,
                 coupling_method = 'uniform',
                 model="mlp",
                 device="cuda:0",
                 n_trials=100,
                 epochs=500,
                 batch_size=(5, 50),
                 learning_rate=(1e-5, 1e-2), 
                 hidden_dim=(16, 512), 
                 num_layers=(1, 5),
                 activation=("ReLU", "LeakyReLU"),
                 time_embed_dim=(8, 64), 
                 dropout=(0.0, 0.5),
                 gamma=(0.0, 2.0),
                 num_timesteps=100,
                 metrics=[MetricsAvaliable.mse_histograms,
                          MetricsAvaliable.mnist_plot,
                          MetricsAvaliable.marginal_binary_histograms]):

        #...params
        self.dynamics = dynamics
        self.experiment_type = experiment_type + "_" + model + "_" + str(datetime.datetime.now().strftime("%Y.%m.%d_%Hh%Ms%S"))
        self.experiment_indentifier = experiment_indentifier
        self.workdir = "/home/df630/conditional_rate_matching/results/{}/{}".format(dynamics, self.experiment_type)
        self.dataset1 = dataset1
        self.model = model
        self.thermostat = thermostat
        self.coupling_method = coupling_method
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = list(activation)
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.gamma = gamma
        self.num_timesteps = num_timesteps
        self.metrics = metrics

        #...scan
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

        #...scaning params:
        epochs = self.def_param(trial, 'epochs', self.epochs, type="int")
        batch_size = self.def_param(trial, 'bach_size', self.batch_size, type="int")
        learning_rate = self.def_param(trial, 'lr', self.learning_rate, type="float")
        hidden_dim = self.def_param(trial, 'dim_hid', self.hidden_dim, type="int")
        num_layers = self.def_param(trial, 'num_layers', self.num_layers, type="int")
        activation = self.def_param(trial, 'activation', self.activation, type="cat")
        time_embed_dim = self.def_param(trial, 'dim_t_emb', self.time_embed_dim, type="int")
        dropout = self.def_param(trial, 'dropout', self.dropout, type="float")
        gamma = self.def_param(trial, 'gamma', self.gamma, type="float") if self.gamma is not None else None

        #...run single experiment:

        metrics = CRM_single_run(dynamics=self.dynamics,
                                experiment_type=self.experiment_type,
                                experiment_indentifier=exp_id,
                                model=self.model,
                                thermostat=self.thermostat,
                                coupling_method=self.coupling_method,
                                dataset1=self.dataset1,
                                metrics=self.metrics,
                                device=self.device,
                                epochs=epochs,
                                batch_size=batch_size,
                                learning_rate=learning_rate, 
                                hidden_dim=hidden_dim, 
                                num_layers=num_layers,
                                activation=activation,
                                time_embed_dim=time_embed_dim,
                                dropout=dropout,
                                gamma=gamma,
                                num_timesteps=self.num_timesteps)
        

        print('all metric: ', metrics)
        self.nist_metric = metrics["mse_marginal_histograms"]
        if self.nist_metric < self.metric: self.metric = self.nist_metric
        else: os.system("rm -rf {}/{}".format(self.workdir, exp_id))
        
        return self.nist_metric



if __name__ == "__main__":

    CRM_single_run(dynamics="crm",
                   experiment_type="mnist_LogThermostat",
                   model="convnet",
                   epochs=10,
                   thermostat="log",
                   coupling_method='uniform',
                   dataset1="mnist",
                   batch_size=128,
                   learning_rate=1e-4, 
                   device="cuda:3",
                   dropout=0.25,
                   num_timesteps=250)

    # scan = CRM_Scan_Optuna(dynamics="crm",
    #                        experiment_type="enist_LogThermostat",
    #                        experiment_indentifier="optuna_scan_trial",
    #                        dataset1="mnist",
    #                        thermostat="log",
    #                        coupling_method='uniform',
    #                        model="mlp",
    #                        n_trials=100,
    #                        epochs=5000,
    #                        batch_size=256,
    #                        learning_rate=(1e-6, 1e-2), 
    #                        hidden_dim=(32, 256), 
    #                        num_layers=(2, 6),
    #                        activation=('ReLU', 'Sigmoid', 'SELU'),
    #                        time_embed_dim=(32, 256), 
    #                        dropout=(0.01, 0.5),
    #                        gamma=None,
    #                        device='cuda:2')
    
    # df = scan.study.trials_dataframe()
    # df.to_csv(scan.workdir + '/trials.tsv', sep='\t', index=False)

    # # Save Optimization History
    # fig = plot_optimization_history(scan.study)
    # fig.write_image(scan.workdir + "/optimization_history.png")

    # # Save Slice Plot
    # fig = plot_slice(scan.study)
    # fig.write_image(scan.workdir + "/slice_plot.png")

    # # Save Contour Plot
    # fig = plot_contour(scan.study)
    # fig.write_image(scan.workdir + "/contour_plot.png")

    # # Save Parallel Coordinate Plot
    # fig = plot_parallel_coordinate(scan.study)
    # fig.write_image(scan.workdir + "/parallel_coordinate.png")

    # # Save Parameter Importances
    # fig = plot_param_importances(scan.study)
    # fig.write_image(scan.workdir + "/param_importances.png")