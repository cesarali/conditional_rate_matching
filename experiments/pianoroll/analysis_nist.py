
import argparse
from utils import run_nist_analysis
import os
import torch
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching import data_path
from conditional_rate_matching.models.metrics.metrics_utils import log_metrics, MetricsAvaliable

image_data_path = os.path.join(data_path,"raw")
metrics_avaliable = MetricsAvaliable()
experiment_dir = "/home/df630/conditional_rate_matching/results/test_piano_roll_transformer/crm/1716182222"

device ='cuda:0'
crm = CRM(experiment_dir=experiment_dir, image_data_path=image_data_path, device=device)
crm.config.pipeline.number_of_steps = 10
crm.config.trainer.device = device
crm.config.pipeline.device = device


parser = argparse.ArgumentParser(description='Run MNIST results analysis')
parser.add_argument('--experiment_dir', type=str, required=True, help='Experiment directory')
parser.add_argument('--experiment_name', type=str, required=True, help='Experiment to analyze')
parser.add_argument('--results_dir', type=str, required=False, help='where to store results', default=None)
parser.add_argument('--overwrite', type=str, required=False, help='Overwrite existing analysis', default='False')
parser.add_argument('--num_timesteps', type=int, required=False, help='Number of generation time-steps', default=100)
parser.add_argument('--timepsilon', type=float, required=False, help='Stop at time t=1-epsilon from target', default=None)        
parser.add_argument('--device', type=str, required=False, help='GPU device', default='cuda:0')

arg = parser.parse_args()

overwrite = True if arg.overwrite.lower() in ['true', '1', 't', 'y', 'yes'] else False

run_nist_analysis(os.path.join(arg.experiment_dir, arg.experiment_name),
                  arg.experiment_name,
                  num_timesteps=arg.num_timesteps,
                  time_epsilon=arg.timepsilon,
                  num_img_bridge=10, 
                  num_intermediate_bridge=20,
                  device=arg.device,
                  overwrite=overwrite)
