from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig, LogThermostatConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import SequenceTransformerConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig

""" Default configurations for training.
"""

def CRM_single_run(dynamics="crm",
                    experiment_type="pianoroll",
                    experiment_indentifier="run",
                    thermostat=None,
                    coupling_method = 'uniform', # uniform, OTPlanSampler
                    model="transformer",
                    dataset0=None,
                    dataset1="mnist",
                    metrics=[MetricsAvaliable.mse_histograms,
                             MetricsAvaliable.mnist_plot,
                             MetricsAvaliable.fid_nist,
                             MetricsAvaliable.marginal_binary_histograms],
                    device="cpu",
                    epochs=100,
                    batch_size=32,
                    learning_rate=1e-3, 
                    hidden_dim=256,
                    time_embed_dim=128,
                    dropout=0.1,
                    num_layers=3,
                    activation="ReLU",
                    gamma=1.0,
                    thermostat_time_exponential=3.,
                    thermostat_time_base=1.0,
                    num_timesteps=1000,
                    ema_decay=0.999
                    ):

    experiment_files = ExperimentFiles(experiment_name=dynamics,
                                       experiment_type=experiment_type,
                                       experiment_indentifier=experiment_indentifier,
                                       delete=True)
    #...configs:

    crm_config = CRMConfig()

    if dataset0 is None:
        crm_config.data0 = StatesDataloaderConfig(dirichlet_alpha=100., batch_size=batch_size)
        crm_config.data0.max_test_size = 50

    crm_config.data1 = crm_config.data0

    if model=="mlp":
        pass
    elif model=="transformer":
        pass

    if thermostat == "log": crm_config.thermostat = LogThermostatConfig(time_exponential=thermostat_time_exponential, time_base=thermostat_time_base,)
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

    print('metrics=',metrics)
    return metrics


if __name__ == "__main__":

        CRM_single_run(dynamics="crm",
               experiment_type="fashion_to_mnist_OT_unet_cfm",
               model="unet_cfm",
               epochs=100,
               thermostat=None,
               coupling_method='uniform',
               dataset0="fashion",
               dataset1="mnist",
               metrics = ["mse_histograms", 
                          'fid_nist', 
                          "mnist_plot", 
                          "marginal_binary_histograms"],
               batch_size=256,
               learning_rate= 0.0001,
               hidden_dim=128,
               time_embed_dim=128,
               gamma=1,
               device="cuda:2")