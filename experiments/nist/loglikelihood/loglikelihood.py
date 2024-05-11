import os
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.metrics.crm_loglikelihood import get_log_likelihood
from conditional_rate_matching import results_path
import argparse


parser = argparse.ArgumentParser(description='Run MNIST Liklihood Estimation')
parser.add_argument('--fwd_process', type=str, required=True, help='fwd process source -> target')
parser.add_argument('--bck_process', type=str, required=True, help='bck process target -> source')
parser.add_argument('--timesteps', type=int, required=False, help='number of timesteps', default=100)
arg = parser.parse_args()

device = "cuda:0"

fwd_dir = os.path.join(results_path, "crm", "images", arg.fwd_process, "run")
bck_dir = os.path.join(results_path, "crm", "images", arg.bck_process, "run")
crm_fwd = CRM(experiment_dir=fwd_dir, device=device)
crm_bck = CRM(experiment_dir=bck_dir, device=device)

NLL = get_log_likelihood(crm_fwd, crm_bck, in_batches=True, device=device)
print(NLL)

