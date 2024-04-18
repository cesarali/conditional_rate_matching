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

from utils import run_nist_analysis

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
                    thermostat_params={'gamma': 1.0, 
                                       'max': 1.0, 
                                       'slope': 1.0, 
                                       'shift': 0.65, 
                                       'exponent': 1.0,
                                       'log_exponential': 0.1,
                                       'time_base': 1.0},
                    num_timesteps=100,
                    ema_decay=0.999,
                    run_analysis=True,
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

    if thermostat == "LogThermostat": crm_config.thermostat = LogThermostatConfig(time_exponential=thermostat_params['log_exponential'], time_base=thermostat_params['time_base'],)
    elif thermostat == "ExponentialThermostat": crm_config.thermostat = ExponentialThermostatConfig(max=thermostat_params['max'], gamma=thermostat_params['gamma'],)
    elif thermostat == "InvertedExponentialThermostat": crm_config.thermostat = InvertedExponentialThermostatConfig(max=thermostat_params['max'], gamma=thermostat_params['gamma'],)
    elif thermostat == "PeriodicThermostat": crm_config.thermostat = PeriodicThermostatConfig(max=thermostat_params['max'], gamma=thermostat_params['gamma'],)
    elif thermostat == "PolynomialThermostat": crm_config.thermostat = PolynomialThermostatConfig(gamma=thermostat_params['gamma'], exponent=thermostat_params['exponent'])
    elif thermostat == "PlateaulThermostat": crm_config.thermostat = PlateauThermostatConfig(gamma=thermostat_params['gamma'], slope=thermostat_params['slope'], shift=thermostat_params['shift'])
    else: crm_config.thermostat = ConstantThermostatConfig(gamma=thermostat_params['gamma'])

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

    print('INFO: final metrics = ', metrics)

    if run_analysis:
       print('INFO: running analysis')
       run_nist_analysis(experiment_type,
                        run=experiment_indentifier,
                        generative_model=dynamics,
                        num_timesteps=num_timesteps,
                        time_epsilon=time_embed_dim,
                        num_img_bridge=6, 
                        num_intermediate_bridge=5,
                        device=device)
    return metrics


if __name__ == "__main__":

    #..Parse the arguments

    import argparse
    import random

    parser = argparse.ArgumentParser(description='Run the CRM training with specified configurations.')
    parser.add_argument('--cuda', type=str, required=True, help='CUDA device number')
    parser.add_argument('--source', type=str, required=True, help='Source dataset')
    parser.add_argument('--target', type=str, required=True, help='Target dataset')
    parser.add_argument('--model', type=str, required=True, help='Model for the network')
    parser.add_argument('--timesteps', type=int, required=False, help='Number of timesteps', default=100)
    parser.add_argument('--coupling', type=str, required=False, help='Type of source-target coupling', default='uniform')
    parser.add_argument('--epochs', type=int, required=False, help='Number of epochs', default=100)
    parser.add_argument('--dim', type=int, required=False, help='Hidden dimension size', default=145)
    parser.add_argument('--act', type=str, required=False, help='Activation function', default='Swish')
    parser.add_argument('--lr', type=float, required=False, help='Learning rate', default=0.0004)
    parser.add_argument('--ema', type=float, required=False, help='Exponential moving average decay', default=0.9999)
    parser.add_argument('--thermostat', type=str, required=True, help='Type of thermostat', default='Constant')
    parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for thermostat', default=None)
    parser.add_argument('--max', type=float, required=False, help='Max parameter for thermostat', default=None)
    parser.add_argument('--slope', type=float, required=False, help='Slope parameter for thermostat', default=None)
    parser.add_argument('--shift', type=float, required=False, help='Shift parameter for thermostat', default=None)
    parser.add_argument('--exponent', type=float, required=False, help='Exponent parameter for thermostat', default=None)
    parser.add_argument('--logexp', type=float, required=False, help='Exponential parameter for thermostat', default=None)
    parser.add_argument('--timebase', type=float, required=False, help='Time base parameter for thermostat', default=None)
    args = parser.parse_args()

    params = ['gamma', 'max', 'slope', 'shift', 'exponent', 'logexp', 'timebase']
    therm_params = {}
    for p in params:
        if p in vars(args).keys():
            if vars(args)[p] is not None:
                therm_params[p] = vars(args)[p]

    # Determine dataset0 based on experiment name
    import datetime
    date = datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    full_experiment_type = f"{args.source}_to_{args.target}_{args.model}_dim_{args.dim}_{args.act}_{args.thermostat}Thermostat" + "_" + "_".join([f"{k}_{v}" for k,v in therm_params.items()]) + "__" + date 


    # Call the CRM_single_run function with the parsed arguments
    CRM_single_run(dynamics="crm",
                   experiment_type=full_experiment_type,
                   model=args.model,
                   epochs=args.epochs,
                   thermostat=args.thermostat + "Thermostat",
                   coupling_method=args.coupling,
                   dataset0=None if args.source == 'noise' else args.source,
                   dataset1=args.target,
                   metrics=["mse_histograms", 
                            "fid_nist", 
                            "mnist_plot",
                            "marginal_binary_histograms"],
                   batch_size=256,
                   learning_rate=args.lr,
                   ema_decay=args.ema,
                   hidden_dim=args.dim,
                   time_embed_dim=args.dim,
                   thermostat_params=therm_params,
                   activation=args.act,
                   num_timesteps=args.timesteps,
                   device="cuda:" + args.cuda)


    #     # import sys

    #     # cuda = sys.argv[1]
    #     # experiment = sys.argv[2]
    #     # dim_hidden = int(sys.argv[3])
    #     # act = sys.argv[4]
    #     # thermostat = sys.argv[5] + "Thermostat"
    #     # gamma = sys.argv[6]
    #     # max = sys.argv[7]
    #     # dataset0 = experiment.split('_')[0]

    #     # print('experiment=',experiment, 'thermostat=',thermostat, 'gamma=',gamma, 'max=',max, 'dataset0=',dataset0, 'cuda=',cuda)

    #     # if dataset0 == 'noise': dataset0 = None
    #     # coupling = 'OTPlanSampler' if experiment.split('_')[-1] == 'OT' else 'uniform'

    #     # CRM_single_run(dynamics="crm",
    #     #        experiment_type=experiment + '_hiddim_' + str(dim_hidden) + '_' + act  +'_' + thermostat + '_gamma_' + gamma + '_max_' + max,
    #     #        model="unet",
    #     #        epochs=2,
    #     #        thermostat=thermostat+"Thermostat",
    #     #        coupling_method=coupling,
    #     #        dataset0=dataset0,
    #     #        dataset1="mnist",
    #     #        metrics = ["mse_histograms", 
    #     #                   'fid_nist', 
    #     #                   "mnist_plot", 
    #     #                   "marginal_binary_histograms"],
    #     #        batch_size=256,
    #     #        learning_rate=0.0004,
    #     #        ema_decay=0.9999,
    #     #        hidden_dim=dim_hidden,
    #     #        time_embed_dim=dim_hidden,
    #     #        gamma=float(gamma),
    #     #        max=float(max),
    #     #        activation=act,
    #     #        num_timesteps=10,
    #     #        device="cuda:" + cuda)
