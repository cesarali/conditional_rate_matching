
import argparse
from utils import run_nist_analysis
import os
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
