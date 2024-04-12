import os
import torch
from conditional_rate_matching import results_path
from utils import generate_mnist_samples, mnist_classifier, mnist_noise_bridge, mnist_grid, get_fid, get_nist_metrics

#  "mnist/uniform_coupling/noise_to_mnist_unet_cfm_1.0"
#  "noise_to_mnist_unet_ConstantThermostat_gamma_0.05_max_0.0"
#  "emnist_to_mnist_unet_cfm_1000epochs_1.0_0"
#-----
generative_model = "ctdd"
path = "noise_to_mnist"
run = "run"
num_timesteps = 100
time_epsilon = 0.01
device = "cuda:0"

#-----

experiment_dir = os.path.join(results_path, generative_model, path, run)

if not os.path.isfile(experiment_dir + "/sample_gen_x1.dat"):
    x_0, x_1, x_test = generate_mnist_samples(path=experiment_dir,  
                                              generative_model=generative_model,
                                              num_timesteps=num_timesteps,
                                              time_epsilon=time_epsilon,  
                                              device=device)
else:
    x_0 = torch.load(experiment_dir + "/sample_gen_x0.dat")
    x_1 = torch.load(experiment_dir + "/sample_gen_x1.dat")
    x_test = torch.load(experiment_dir + "/sample_gen_test.dat")

print(x_test.shape, x_1.shape, x_0.shape)

mnist_grid(x_1[:100], save_path=experiment_dir, num_img=100, nrow=10, figsize=(4, 4))
mnist_classifier(x_1, save_path=experiment_dir, plot_histogram=True)

if not os.path.isfile(experiment_dir + "/bridge_example.png") and generative_model == "crm":
    mnist_noise_bridge(experiment_dir,
                       x_0, 
                       num_timesteps=num_timesteps,  
                       time_epsilon=time_epsilon,
                       num_img=6, 
                       num_timesteps_displayed=20, 
                       save_path=experiment_dir) 

get_fid(x_1, x_test, experiment_dir)
get_nist_metrics(x_1, x_test, experiment_dir)

