from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig, TemporalNetworkToRateConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (ConstantThermostatConfig, 
                                                                                         LogThermostatConfig, 
                                                                                         ExponentialThermostatConfig,  
                                                                                         InvertedExponentialThermostatConfig, 
                                                                                         PeriodicThermostatConfig,
                                                                                         PolynomialThermostatConfig,
                                                                                         PlateauThermostatConfig)

from conditional_rate_matching.models.temporal_networks.temporal_networks_config import SequenceTransformerConfig, TemporalDeepMLPConfig


from utils import run_nist_analysis

""" Default configurations for training.
"""

def CRM_single_run(experiment_name="pianoroll",
                   experiment_dir=None,
                   thermostat=None,
                   coupling_method='uniform', # uniform, OTl2, OTlog
                   model="unet",
                   dataset0="noise",
                   dataset1="music",
                   metrics=[],
                   device="cpu",
                   epochs=100,
                   batch_size=256,
                   learning_rate=2e-4, 
                   hidden_dim=256,
                   time_embed_dim=128,
                   dropout=0.1,
                   num_layers=6,
                   activation="Swish",
                   thermostat_params={'gamma': 1.0, 
                                        'max': 1.0, 
                                        'slope': 1.0, 
                                        'shift': 0.65, 
                                        'exponent': 1.0,
                                        'log_exponential': 0.1,
                                        'time_base': 1.0},
                   num_timesteps=100,
                   time_epsilon=0.0,
                   ema_decay=0.9999,
                   run_analysis=True,
                   temp_to_rate=None # bernoulli, empty, linear,logistic, None
                    ):

    experiment_files = ExperimentFiles(experiment_dir=experiment_dir,
                                       experiment_type=experiment_name,
                                       delete=False)
    #...configs:

    crm_config = CRMConfig()

    crm_config.data0 = LakhPianoRollConfig(batch_size=batch_size,
                                           conditional_model=True,
                                           bridge_conditional=False)
        
    crm_config.data1 = crm_config.data0

    if model=="mlp": crm_config.temporal_network=TemporalDeepMLPConfig(hidden_dim=hidden_dim, time_embed_dim=time_embed_dim, num_layers=num_layers, activation=activation, dropout=dropout)
    if model=="transformer": crm_config.temporal_network=SequenceTransformerConfig()

    if thermostat=="LogThermostat": crm_config.thermostat = LogThermostatConfig(time_exponential=thermostat_params['log_exponential'], time_base=thermostat_params['time_base'],)
    elif thermostat=="ExponentialThermostat": crm_config.thermostat = ExponentialThermostatConfig(max=thermostat_params['max'], gamma=thermostat_params['gamma'],)
    elif thermostat=="InvertedExponentialThermostat": crm_config.thermostat = InvertedExponentialThermostatConfig(max=thermostat_params['max'], gamma=thermostat_params['gamma'],)
    elif thermostat=="PeriodicThermostat": crm_config.thermostat = PeriodicThermostatConfig(max=thermostat_params['max'], gamma=thermostat_params['gamma'],)
    elif thermostat=="PolynomialThermostat": crm_config.thermostat = PolynomialThermostatConfig(gamma=thermostat_params['gamma'], exponent=thermostat_params['exponent'])
    elif thermostat=="PlateaulThermostat": crm_config.thermostat = PlateauThermostatConfig(gamma=thermostat_params['gamma'], slope=thermostat_params['slope'], shift=thermostat_params['shift'])
    else: crm_config.thermostat=ConstantThermostatConfig(gamma=thermostat_params['gamma'])

    crm_config.trainer = CRMTrainerConfig(number_of_epochs=epochs,
                                          learning_rate=learning_rate,
                                          device=device,
                                          metrics=metrics,
                                          loss_regularize_square=False,
                                          loss_regularize=False,
                                          save_model_epochs=1e6,)
    
    crm_config.pipeline.number_of_steps = num_timesteps
    crm_config.optimal_transport.name = 'OTPlanSampler' if 'OT' in coupling_method else 'uniform'
    crm_config.optimal_transport.cost = 'log' if coupling_method == 'OTlog' else None
    crm_config.temporal_network_to_rate = TemporalNetworkToRateConfig(type_of=temp_to_rate, linear_reduction=None)

    #...train

    crm = CRMTrainer(crm_config, experiment_files)
    _, metrics = crm.train()

    if run_analysis:
       print('INFO: running analysis')
       run_nist_analysis(experiment_dir=experiment_dir,
                         experiment_name=experiment_name,
                         num_timesteps=num_timesteps,
                         time_epsilon=time_epsilon,
                         num_img_bridge=10, 
                         num_intermediate_bridge=20,
                         device=device)
    return metrics


if __name__ == "__main__":

    import argparse
    import datetime
    import os
    import torch
    from conditional_rate_matching import results_path

    date = datetime.datetime.now().strftime("%Hh%Ms%S_%Y.%m.%d")

    if torch.cuda.is_available():

        #..Parse the arguments

        parser = argparse.ArgumentParser(description='Run the CRM training with specified configurations.')

        parser.add_argument('--source', type=str, required=True, help='Source dataset')
        parser.add_argument('--target', type=str, required=True, help='Target dataset')
        parser.add_argument('--model', type=str, required=True, help='Model for the network')
        parser.add_argument('--gamma', type=float, required=True, help='Gamma parameter for thermostat')

        parser.add_argument('--results_dir', type=str, required=False, help='where to store results', default=None)
        parser.add_argument('--id', type=str, required=False, help='Experiment indentifier', default='run')
        parser.add_argument('--timesteps', type=int, required=False, help='Number of timesteps', default=100)
        parser.add_argument('--timepsilon', type=float, required=False, help='Stop at time t=1-epsilon from target', default=None)        
        parser.add_argument('--coupling', type=str, required=False, help='Type of source-target coupling', default='uniform')
        parser.add_argument('--epochs', type=int, required=False, help='Number of epochs', default=100)
        parser.add_argument('--batch_size', type=int, required=False, help='Batch size', default=256)
        parser.add_argument('--dim', type=int, required=False, help='Hidden dimension size', default=145)
        parser.add_argument('--temp_to_rate', type=str, required=False, help='Type of temporal to rate head for network', default=None)
        parser.add_argument('--act', type=str, required=False, help='Activation function', default='Swish')
        parser.add_argument('--lr', type=float, required=False, help='Learning rate', default=0.0004)
        parser.add_argument('--ema', type=float, required=False, help='Exponential moving average decay', default=0.9999)
        parser.add_argument('--thermostat', type=str, required=False, help='Type of thermostat', default='Constant')
        parser.add_argument('--max', type=float, required=False, help='Max parameter for thermostat', default=None)
        parser.add_argument('--slope', type=float, required=False, help='Slope parameter for thermostat', default=None)
        parser.add_argument('--shift', type=float, required=False, help='Shift parameter for thermostat', default=None)
        parser.add_argument('--exponent', type=float, required=False, help='Exponent parameter for thermostat', default=None)
        parser.add_argument('--logexp', type=float, required=False, help='Exponential parameter for thermostat', default=None)
        parser.add_argument('--timebase', type=float, required=False, help='Time base parameter for thermostat', default=None)
        parser.add_argument('--device', type=str, required=False, help='Selected device', default="cuda:0")

        args = parser.parse_args()

        params = ['gamma', 'max', 'slope', 'shift', 'exponent', 'logexp', 'timebase']
        therm_params = {}
        for p in params:
            if p in vars(args).keys():
                if vars(args)[p] is not None:
                    therm_params[p] = vars(args)[p]

        experiment_name = f"{args.source}_to_{args.target}_{args.model}_{args.coupling}_coupling_{args.thermostat}Thermostat" + "_" + "_".join([f"{k}_{v}" for k,v in therm_params.items()]) + "__" + date + '__' + args.id
        
        if args.results_dir == 'amarel': args.results_dir =  os.path.join('/scratch', 'df630', 'conditional_rate_matching', 'results', 'crm', 'images', experiment_name)
        elif args.results_dir == 'nercs': args.results_dir = os.path.join('/scratch', experiment_name)
        else: args.results_dir = os.path.join(results_path, 'crm', 'images', experiment_name)
       
        CRM_single_run(experiment_name=experiment_name,
                       experiment_dir=args.results_dir,
                       model=args.model,
                       thermostat=args.thermostat + "Thermostat",
                       thermostat_params=therm_params,
                       coupling_method=args.coupling,
                       dataset0=args.source,
                       dataset1=args.target,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       learning_rate=args.lr,
                       ema_decay=args.ema,
                       hidden_dim=args.dim,
                       time_embed_dim=args.dim,
                       temp_to_rate=args.temp_to_rate,
                       activation=args.act,
                       num_timesteps=args.timesteps,
                       device=args.device)

    else:
        print("ERROR: CUDA is not available")
        exit(1)