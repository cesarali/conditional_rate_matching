from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (ConstantThermostatConfig, 
                                                                                         LogThermostatConfig, 
                                                                                         ExponentialThermostatConfig,  
                                                                                         InvertedExponentialThermostatConfig, 
                                                                                         PeriodicThermostatConfig,
                                                                                         PolynomialThermostatConfig,
                                                                                         PlateauThermostatConfig)

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import (UConvNISTNetConfig, 
                                                                                         TemporalDeepMLPConfig, 
                                                                                         TemporalLeNet5Config, 
                                                                                         TemporalLeNet5AutoencoderConfig, 
                                                                                         TemporalUNetConfig, 
                                                                                         CFMUnetConfig)
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig

""" Default configurations for training.
"""

def CRM_single_run(dynamics="crm",
                    experiment_type="nist",
                    experiment_indentifier="run",
                    thermostat=None,
                    coupling_method = 'uniform', # uniform, OTPlanSampler
                    model="unet_conv",
                    dataset0="fashion",
                    dataset1="mnist",
                    metrics=[MetricsAvaliable.mse_histograms,
                             MetricsAvaliable.mnist_plot,
                             MetricsAvaliable.fid_nist,
                             MetricsAvaliable.marginal_binary_histograms],
                    device="cpu",
                    epochs=100,
                    batch_size=64,
                    learning_rate=1e-3, 
                    hidden_dim=256,
                    time_embed_dim=128,
                    dropout=0.1,
                    num_layers=3,
                    activation="GELU",
                    gamma =1.0,
                    max=1.0,
                    num_timesteps=50,
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

    if model=="mlp":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=True, as_image=False, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=True, as_image=False, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation,
                                                            dropout = dropout)

    if model=="lenet5":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = TemporalLeNet5Config(hidden_dim = hidden_dim,
                                                           time_embed_dim = time_embed_dim,
                                                           ema_decay=ema_decay)

    if model=="lenet5Autoencoder":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = TemporalLeNet5AutoencoderConfig(hidden_dim = hidden_dim,
                                                                      time_embed_dim = time_embed_dim,
                                                                      ema_decay=ema_decay)

    if model=="unet":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = TemporalUNetConfig(hidden_dim = hidden_dim,
                                                         time_embed_dim = hidden_dim,
                                                         ema_decay=ema_decay,
                                                         activation = activation,
                                                         dropout = dropout)

    if model=="unet_cfm":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = CFMUnetConfig()


    if model=="unet_conv":
        if dataset0 is not None:
            crm_config.data0 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset0)
        crm_config.data1 = NISTLoaderConfig(flatten=False, as_image=True, batch_size=batch_size, dataset_name=dataset1)
        crm_config.temporal_network = UConvNISTNetConfig()

    if thermostat == "LogThermostat": crm_config.thermostat = LogThermostatConfig(time_exponential=gamma, time_base=1.0,)
    elif thermostat == "ExponentialThermostat": crm_config.thermostat = ExponentialThermostatConfig(max=max, gamma=gamma,)
    elif thermostat == "InvertedExponentialThermostat": crm_config.thermostat = InvertedExponentialThermostatConfig(max=max, gamma=gamma,)
    elif thermostat == "PeriodicThermostat": crm_config.thermostat = PeriodicThermostatConfig(max=max, gamma=gamma,)
    elif thermostat == "PolynomialThermostat": crm_config.thermostat = PolynomialThermostatConfig(gamma=gamma, exponent=max)
    elif thermostat == "PlateaulThermostat": crm_config.thermostat = PlateauThermostatConfig(gamma=gamma, slope=max, shift=0.65)

    else: 
        crm_config.thermostat = ConstantThermostatConfig(gamma=gamma)

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
        
        import sys

        cuda = sys.argv[1]
        experiment = sys.argv[2]
        dim_hidden = int(sys.argv[3])
        act = sys.argv[4]
        thermostat = sys.argv[5] + "Thermostat"
        gamma = sys.argv[6]
        max = sys.argv[7]
        dataset0 = experiment.split('_')[0]

        print('experiment=',experiment, 'thermostat=',thermostat, 'gamma=',gamma, 'max=',max, 'dataset0=',dataset0, 'cuda=',cuda)

        if dataset0 == 'noise': dataset0 = None
        coupling = 'OTPlanSampler' if experiment.split('_')[-1] == 'OT' else 'uniform'

        CRM_single_run(dynamics="crm",
               experiment_type=experiment + '_hiddim_' + str(dim_hidden) + '_' + act  +'_' + thermostat + '_gamma_' + gamma + '_max_' + max,
               model="unet",
               epochs=10,
               thermostat=thermostat+"Thermostat",
               coupling_method=coupling,
               dataset0=dataset0,
               dataset1="mnist",
               metrics = ["mse_histograms", 
                          'fid_nist', 
                          "mnist_plot", 
                          "marginal_binary_histograms"],
               batch_size=256,
               learning_rate=0.0004,
               ema_decay=0.9999,
               hidden_dim=dim_hidden,
               time_embed_dim=dim_hidden,
               gamma=float(gamma),
               max=float(max),
               activation=act,
               num_timesteps=200,
               device="cuda:" + cuda)
