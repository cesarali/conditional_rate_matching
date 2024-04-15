from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig, CRMTrainerConfig
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import ConstantThermostatConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalUNetConfig, TemporalDeepMLPConfig
from conditional_rate_matching.models.metrics.metrics_utils import MetricsAvaliable
from conditional_rate_matching.data.image_dataloaders_conditional_config import DistortedNISTLoaderConfig

""" Default configurations for training.
"""

def CRM_single_run(dynamics="crm",
                    experiment_type="distorted_nist",
                    experiment_indentifier="run",
                    thermostat=None,
                    coupling_method = 'uniform',
                    model="mlp",
                    dataset1="mnist",
                    distortion="noise",
                    distortion_level=0.4,
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
                    activation="ReLU",
                    max = 0.0,
                    gamma=1.0,
                    time_epsilon=1e-3,
                    num_timesteps=1000,
                    ema_decay=0.999
                    ):

    experiment_files = ExperimentFiles(experiment_name=dynamics,
                                       experiment_type=experiment_type,
                                       experiment_indentifier=experiment_indentifier,
                                       delete=True)
    #...configs:

    crm_config = CRMConfig()

    if model=="mlp":
        crm_config.data1 = DistortedNISTLoaderConfig(flatten=True, 
                                                     as_image=False, 
                                                     distortion=distortion,
                                                     distortion_level=distortion_level,
                                                     batch_size=batch_size, 
                                                     dataset_name=dataset1)
        
        crm_config.data0 = crm_config.data1
        crm_config.temporal_network = TemporalDeepMLPConfig(hidden_dim = hidden_dim,
                                                            time_embed_dim = time_embed_dim,
                                                            num_layers = num_layers,
                                                            activation = activation,
                                                            dropout = dropout)

    if model=="unet":
        crm_config.data1 = DistortedNISTLoaderConfig(flatten=False, 
                                                     as_image=True, 
                                                     distortion=distortion,
                                                     distortion_level=distortion_level,
                                                     batch_size=batch_size, 
                                                     dataset_name=dataset1)
                
        crm_config.data0 = crm_config.data1
        crm_config.temporal_network = TemporalUNetConfig(hidden_dim = hidden_dim,
                                                         time_embed_dim = hidden_dim,
                                                         ema_decay=ema_decay)


    crm_config.thermostat = ConstantThermostatConfig(gamma=gamma)

    crm_config.trainer = CRMTrainerConfig(number_of_epochs=epochs,
                                          learning_rate=learning_rate,
                                          device=device,
                                          metrics=metrics,
                                          loss_regularize_square=False,
                                          loss_regularize=False)
    
    crm_config.pipeline.number_of_steps = num_timesteps
    crm_config.pipeline.time_epsilon = time_epsilon
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
               dataset1="mnist",
               distortion="noise",
               distortion_level=0.5,
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

    # CRM_single_run(dynamics="crm",
    #            experiment_type='noise_mnist_to_mnist_mlp_0.01',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="noise",
    #            distortion_level=0.4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.01,
    #            device="cuda:2")

    # CRM_single_run(dynamics="crm",
    #            experiment_type='noise_mnist_to_mnist_mlp_0.001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="noise",
    #            distortion_level=0.4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.001,
    #            device="cuda:2")
    
    # CRM_single_run(dynamics="crm",
    #            experiment_type='noise_mnist_to_mnist_mlp_0.0001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="noise",
    #            distortion_level=0.4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.0001,
    #            device="cuda:2")

    # CRM_single_run(dynamics="crm",
    #            experiment_type='noise_mnist_to_mnist_mlp_0.00001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="noise",
    #            distortion_level=0.4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.00001,
    #            device="cuda:2")
    


    # CRM_single_run(dynamics="crm",
    #            experiment_type='swirl_mnist_to_mnist_mlp_0.01',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="swirl",
    #            distortion_level=4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.01,
    #            device="cuda:2")

    # CRM_single_run(dynamics="crm",
    #            experiment_type='swirl_mnist_to_mnist_mlp_0.001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="swirl",
    #            distortion_level=4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.001,
    #            device="cuda:2")
    
    # CRM_single_run(dynamics="crm",
    #            experiment_type='swirl_mnist_to_mnist_mlp_0.0001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="swirl",
    #            distortion_level=4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.0001,
    #            device="cuda:2")

    # CRM_single_run(dynamics="crm",
    #            experiment_type='swirl_mnist_to_mnist_mlp_0.00001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='uniform',
    #            dataset1="mnist",
    #            distortion="swirl",
    #            distortion_level=4,
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.00001,
    #            device="cuda:2")

    
    # CRM_single_run(dynamics="crm",
    #            experiment_type='halfmask_mnist_to_mnist_OT_mlp_0.001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='OTPlanSampler',
    #            dataset1="mnist",
    #            distortion="half_mask",
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.001,
    #            device="cuda:1")

    # CRM_single_run(dynamics="crm",
    #            experiment_type='halfmask_mnist_to_mnist_OT_mlp_0.0001',
    #            model="mlp",
    #            epochs=100,
    #            thermostat=None,
    #            coupling_method='OTPlanSampler',
    #            dataset1="mnist",
    #            distortion="half_mask",
    #            metrics = ["mse_histograms", 
    #                       'fid_nist', 
    #                       "mnist_plot", 
    #                       "marginal_binary_histograms"],
    #            batch_size=256,
    #            learning_rate= 0.0001,
    #            hidden_dim=256,
    #            num_layers=7,
    #            dropout=0.15,
    #            time_embed_dim=128,
    #            ema_decay=0.99999,
    #            num_timesteps=1000,
    #            time_epsilon=0.05,
    #            activation="GELU",
    #            gamma=0.0001,
    #            device="cuda:1")