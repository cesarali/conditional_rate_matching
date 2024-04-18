import os
import torch
from conditional_rate_matching import results_path
from utils import run_analysis, generate_mnist_samples, mnist_classifier, mnist_noise_bridge, mnist_grid, get_fid, get_nist_metrics

#  "mnist/uniform_coupling/noise_to_mnist_unet_cfm_1.0"
#  "noise_to_mnist_unet_ConstantThermostat_gamma_0.05_max_0.0"
#  "emnist_to_mnist_unet_cfm_1000epochs_1.0_0"
#-----

device = "cuda:0"


path = "corrupted_to_mnist_unet_att_hiddim_145_Swish_ConstantThermostat_gamma_0.5_max_0.0"
time_epsilon = 0.0
num_timesteps = 200
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

path = "fashion_to_mnist_unet_att_hiddim_145_Swish_PlateauThermostat_gamma_0.5_max_15.0"
time_epsilon = 0.005
num_timesteps = 200
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

path = "fashion_to_mnist_unet_att_hiddim_145_Swish_PlateauThermostat_gamma_1.0_max_15.0"
time_epsilon = 0.005
num_timesteps = 200
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

path = "noise_to_mnist_unet_att_hiddim_145_Swish_PlateauThermostat_gamma_0.5_max_15.0"
time_epsilon = 0.005
num_timesteps = 200
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

path = "noise_to_mnist_unet_att_hiddim_145_Swish_PlateauThermostat_gamma_1.0_max_15.0"
time_epsilon = 0.005
num_timesteps = 200
run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_hiddim_145_Swish_ConstantThermostat_gamma_0.75_max_0.0"
# time_epsilon = 0.01
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_hiddim_145_Swish_ConstantThermostat_gamma_1.0_max_0.0"
# time_epsilon = 0.005
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_hiddim_145_Swish_ConstantThermostat_gamma_1.5_max_0.0"
# time_epsilon = 0.005
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_hiddim_145_Swish_ConstantThermostat_gamma_2.0_max_0.0"
# time_epsilon = 0.005
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_0.01_max_0.0"
# time_epsilon = 0.2
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_0.05_max_0.0"
# time_epsilon = 0.1
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_0.1_max_0.0"
# time_epsilon = 0.05
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_0.25_max_0.0"
# time_epsilon = 0.01
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_0.5_max_0.0"
# time_epsilon = 0.01
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_0.75_max_0.0"
# time_epsilon = 0.01
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_1.0_max_0.0"
# time_epsilon = 0.005
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_1.5_max_0.0"
# time_epsilon = 0.005
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)

# path = "emnist_to_mnist_unet_att_OT_hiddim_145_Swish_ConstantThermostat_gamma_2.0_max_0.0"
# time_epsilon = 0.005
# num_timesteps = 100
# run_analysis(path, num_timesteps=num_timesteps, time_epsilon=time_epsilon, device=device)