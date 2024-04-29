import os
import torch
from conditional_rate_matching import results_path
from utils import run_nist_analysis


experiment = "fashion_to_mnist_unet_uniform_coupling_ConstantThermostat_gamma_0.75__14h12s22_2024.04.29"
device = "cuda:2"

run_nist_analysis(experiment,
                  num_timesteps=100,
                  time_epsilon=0.01,
                  num_img_bridge=6, 
                  num_intermediate_bridge=20,
                  device=device)
