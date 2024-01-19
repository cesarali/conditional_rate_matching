import datetime
import numpy as np
import os
import optuna
from optuna.visualization import plot_optimization_history, plot_slice, plot_contour, plot_parallel_coordinate, plot_param_importances

from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from ctdd_nist_single_run import CTDD_single_run

class CTDD_Scan_Optuna:
    def __init__(self, 
                 dynamics="crm",
                 experiment_type="nist",
                 experiment_indentifier="optuna_scan",
                 dataset0='mnist',
                 model="mlp",
                 device="cuda:0",
                 n_trials=100,
                 epochs=500,
                 batch_size=(5, 50),
                 learning_rate=(1e-5, 1e-2), 
                 hidden_dim=None, 
                 num_layers=None,
                 activation=None,
                 time_embed_dim=(8, 64), 
                 dropout=None,
                 num_timesteps=100,
                 metrics=[MetricsAvaliable.mse_histograms,
                          MetricsAvaliable.fid_nist,
                          MetricsAvaliable.mnist_plot,
                          MetricsAvaliable.marginal_binary_histograms]):

        #...params

        self.dynamics = dynamics
        self.experiment_type = experiment_type + "_" + model + "_" + str(datetime.datetime.now().strftime("%Y.%m.%d_%Hh%Ms%S"))
        self.experiment_indentifier = experiment_indentifier
        self.workdir = "/home/df630/conditional_rate_matching/results/{}/{}".format(dynamics, self.experiment_type)
        self.dataset0 = dataset0
        self.model = model
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
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
        time_embed_dim = self.def_param(trial, 'dim_t_emb', self.time_embed_dim, type="int")

        hidden_dim = self.def_param(trial, 'dim_hid', self.hidden_dim, type="int") if self.hidden_dim is not None else None
        num_layers = self.def_param(trial, 'num_layers', self.num_layers, type="int") if self.num_layers is not None else None
        activation = self.def_param(trial, 'activation', list(self.activation), type="cat") if self.activation is not None else None
        dropout = self.def_param(trial, 'dropout', self.dropout, type="float") if self.dropout is not None else None

        #...run single experiment:

        metrics = CTDD_single_run(dynamics=self.dynamics,
                                  experiment_type=self.experiment_type,
                                  experiment_indentifier=exp_id,
                                  model=self.model,
                                  dataset0=self.dataset0,
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
                                  num_timesteps=self.num_timesteps)
        
        print('all metric: ', metrics)

        mse_histograms = metrics["mse_marginal_histograms"]
        fid_layer_1 = metrics["fid_1"]
        fid_layer_2 = metrics["fid_2"]
        fid_layer_3 = metrics["fid_3"]

        self.nist_metric = (100000 * mse_histograms + fid_layer_1 + fid_layer_2 + fid_layer_3) / 4.0
        if self.nist_metric < self.metric: self.metric = self.nist_metric
        else: os.system("rm -rf {}/{}".format(self.workdir, exp_id))
        
        return self.nist_metric



if __name__ == "__main__":

    # scan = CTDD_Scan_Optuna(dynamics="ctdd",
    #                        experiment_type="mnist",
    #                        experiment_indentifier="optuna_scan_trial",
    #                        dataset0="mnist",
    #                        model="mlp",
    #                        metrics=["fid_nist", 
    #                                 "mse_histograms",  
    #                                 "mnist_plot", 
    #                                 "marginal_binary_histograms"],
    #                        n_trials=250,
    #                        epochs=50,
    #                        batch_size=256,
    #                        learning_rate=(1e-6, 1e-2), 
    #                        num_timesteps=1000,
    #                        hidden_dim=(32, 256),
    #                        time_embed_dim=(32, 256), 
    #                        num_layers=(2, 8),
    #                        dropout=(0.01, 0.5),
    #                        activation=["ReLU", "GELU"],
    #                        device='cuda:1')

    scan = CTDD_Scan_Optuna(dynamics="ctdd",
                           experiment_type="mnist",
                           experiment_indentifier="optuna_scan_trial",
                           dataset0="mnist",
                           model="lenet5",
                           metrics=["fid_nist", 
                                    "mse_histograms",  
                                    "mnist_plot", 
                                    "marginal_binary_histograms"],
                           n_trials=3,
                           epochs=5,
                           batch_size=256,
                           learning_rate=(1e-6, 1e-2), 
                           num_timesteps=1000,
                           hidden_dim=(32, 256),
                           time_embed_dim=(32, 256), 
                           device='cuda:1')

    df = scan.study.trials_dataframe()
    df.to_csv(scan.workdir + '/trials.tsv', sep='\t', index=False)

    # Save Optimization History
    fig = plot_optimization_history(scan.study)
    fig.write_image(scan.workdir + "/optimization_history.png")

    # Save Slice Plot
    fig = plot_slice(scan.study)
    fig.write_image(scan.workdir + "/slice_plot.png")

   # Save Contour Plot
    fig = plot_contour(scan.study)
    fig.write_image(scan.workdir + "/contour_plot.png")

    # Save Parallel Coordinate Plot
    fig = plot_parallel_coordinate(scan.study)
    fig.write_image(scan.workdir + "/parallel_coordinate.png")

    # Save Parameter Importances
    fig = plot_param_importances(scan.study)
    fig.write_image(scan.workdir + "/param_importances.png")