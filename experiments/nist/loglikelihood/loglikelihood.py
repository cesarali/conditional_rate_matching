import os
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.metrics.crm_loglikelihood import get_log_likelihood
from conditional_rate_matching import results_path
import argparse


parser = argparse.ArgumentParser(description='Run MNIST Liklihood Estimation')
parser.add_argument('--fwd_process', type=str, required=True, help='fwd process source -> target')
parser.add_argument('--bck_process', type=str, required=True, help='bck process target -> source')
parser.add_argument('--timesteps', type=int, required=False, help='number of timesteps', default=100)
parser.add_argument('--batchsize', type=int, required=False, help='batch size', default=128)
arg = parser.parse_args()

device = "cuda:0"

fwd_dir = os.path.join(results_path, "crm", "images", arg.fwd_process, "run")
bck_dir = os.path.join(results_path, "crm", "images", arg.bck_process, "run")
crm_fwd = CRM(experiment_dir=fwd_dir, device=device)
crm_bck = CRM(experiment_dir=bck_dir, device=device)

crm_fwd.config.data0.batchsize = arg.batchsize
crm_fwd.config.data1.batchsize = arg.batchsize
crm_bck.config.data0.batchsize = arg.batchsize
crm_bck.config.data1.batchsize = arg.batchsize

get_log_likelihood(crm_fwd, crm_bck, device=device)