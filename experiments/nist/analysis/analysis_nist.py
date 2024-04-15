import os
import torch
from conditional_rate_matching import results_path
from utils import run_analysis, generate_mnist_samples, mnist_classifier, mnist_noise_bridge, mnist_grid, get_fid, get_nist_metrics

#  "mnist/uniform_coupling/noise_to_mnist_unet_cfm_1.0"
#  "noise_to_mnist_unet_ConstantThermostat_gamma_0.05_max_0.0"
#  "emnist_to_mnist_unet_cfm_1000epochs_1.0_0"
#-----

device = "cuda:0"

path = "noise_to_mnist_unet_att__hiddim_145_Swish_ConstantThermostat_gamma_0.75_max_0.0"
time_epsilon = 0.01
num_timesteps = 100
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

path = "noise_to_mnist_unet_att__hiddim_145_GELU_ConstantThermostat_gamma_0.75_max_0.0"
time_epsilon = 0.01
num_timesteps = 100
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

path = "noise_to_mnist_unet__hiddim_256_GELU_ConstantThermostat_gamma_0.75_max_0.0"
time_epsilon = 0.01
num_timesteps = 100
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)
