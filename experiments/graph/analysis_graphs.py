import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping
from conditional_rate_matching import results_path
from torchvision.utils import make_grid

from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_Cifar import experiment_cifar10_config
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (ConstantThermostatConfig, 
                                                                                         PeriodicThermostatConfig,
                                                                                         ExponentialThermostatConfig,
                                                                                         PolynomialThermostatConfig,
                                                                                         PlateauThermostatConfig
                                                                                         )

def generate_samples(path, 
                     x_test,
                     num_timesteps=100, 
                     time_epsilon=0.0,
                     device="cpu"):
    
    crm = CRM(experiment_dir=path, device=device)
    crm.config.pipeline.time_epsilon = time_epsilon
    crm.config.pipeline.num_intermediates = num_timesteps
    crm.config.pipeline.number_of_steps = num_timesteps

    x_1, x_t, t = crm.pipeline(x_test.shape[0], 
                               return_intermediaries=True, 
                               train=False, 
                               x_0=x_test)
    
    x_1 = x_1.view(-1, 3, 32, 32)
    x_t = x_t.view(-1, x_t.shape[1], 3, 32, 32)
    return x_1, x_t, t

def cifar_noise_bridge(path, 
                       x_input, 
                       num_timesteps=1000,  
                       time_epsilon=0.0,
                       num_img=5,
                       num_timesteps_displayed=20,
                       save_path=None):
    
    img_1, img_hist, time_steps = generate_samples(path, x_input[:num_img], num_timesteps=num_timesteps,  time_epsilon=time_epsilon, device=x_input.device)
    _, axs = plt.subplots(num_img, num_timesteps_displayed+2, figsize=(num_timesteps_displayed, num_img))
    N = img_hist.size(1)
    dt = N // num_timesteps_displayed

    img_1 = img_1.long()   
    img_hist = img_hist.long()  

    
    for j, idx in enumerate(np.arange(0, N+1, dt)):
        if j<num_timesteps_displayed:
            tau = time_steps[idx]
            images = img_hist[:, idx, :]
        else:
            tau = time_steps[-1]
            images = img_hist[:, -1, :]
            
        for i in range(num_img):
            img = images[i].detach().cpu().numpy()
            axs[i, j].imshow(img.squeeze().transpose(1, 2, 0) , cmap='gray')
            if i == 0: axs[i, j].set_title(r'$\tau = {0}$'.format(round(tau.item(),2)))
            axs[i, j].axis('off')
    
    j = num_timesteps_displayed + 1
    for i in range(num_img):
        axs[i, j].imshow(img_1[i].detach().cpu().numpy().squeeze().transpose(1, 2, 0) , cmap='gray')
        axs[i, j].axis('off')

    plt.tight_layout()
    if save_path is None: plt.show()
    else: plt.savefig(save_path+'/bridge_example.png')

def cifar_conditional_bridge(source, 
                             target, 
                             thermostat="constant", 
                             thermostat_params=(.1,0),
                             figsize=None, 
                             num_timesteps=100,
                             num_timesteps_displayed=10,
                             save_path=None):
    
    config = experiment_cifar10_config()
    
    if thermostat == 'exponential':
        config.thermostat = ExponentialThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
        config.thermostat.max = thermostat_params[1]
    elif thermostat == 'constant':
        config.thermostat = ConstantThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
    elif thermostat == 'periodic':
        config.thermostat = PeriodicThermostatConfig()
        config.thermostat.gamma= thermostat_params[0]
        config.thermostat.max = thermostat_params[1]
    elif thermostat == 'polynomial':
        config.thermostat = PolynomialThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
        config.thermostat.exponent = thermostat_params[1]
    elif thermostat == 'plateau':
        config.thermostat = PlateauThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
        config.thermostat.slope = thermostat_params[1]
        config.thermostat.shift = thermostat_params[2]

    crm = CRM(config)
    crm.config.pipeline.number_of_steps = num_timesteps
    crm.config.pipeline.num_intermediates = num_timesteps

    rate_model = lambda x, t: crm.forward_rate.conditional_transition_rate(x, target.view(-1, 3*32*32), t)
    img_1, img_hist, _, _ = TauLeaping(crm.config, rate_model, source.view(-1, 3*32*32), forward=True)

    N = img_hist.size(1)
    dt = N // num_timesteps_displayed

    num_img = source.shape[0]
    img_1 = img_1.long().view(-1, 3, 32, 32)
    img_hist = img_hist.long().view(-1, N, 3, 32, 32)
    _, axs = plt.subplots(num_img, num_timesteps_displayed+2, figsize=(num_timesteps_displayed, num_img) if figsize is None else figsize)

    for j, idx in enumerate(np.arange(0, N+1, dt)):
        if j < num_timesteps_displayed:
            images = img_hist[:, idx, :]
        else:
            images = img_hist[:, -1, :]
            
        for i in range(num_img):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Reorder dimensions to (height, width, channels)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
    
    j = num_timesteps_displayed + 1
    for i in range(num_img):
        axs[i, j].imshow(img_1[i].detach().cpu().numpy().transpose(1, 2, 0))  # Adjust this line similarly
        axs[i, j].axis('off')

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path + '/conditional_markov_example.png')



def get_cifar10_test_samples(path, 
                            which = 'source', # 'source' or 'target
                            class_label=None, 
                            sample_size=800, 
                            device="cpu"):
    
    crm = CRM(experiment_dir=path, device=device)
    images = []
    source = crm.dataloader_0.test()
    target = crm.dataloader_1.test()

    for batch in source if which == 'source' else target:
        if len(batch) == 2:
            sample, labels = batch[0], batch[1]
            selected_images = sample[labels == class_label] if class_label is not None else sample 
            images.append(selected_images)
        else:
            sample = batch[0].view(-1, 3, 32, 32)
            images.append(sample)

    return torch.cat(images, dim=0)[:sample_size].to(device) #if labeled else torch.tensor(images, device=device)


def image_grid(sample, save_path='.', num_img=5, nrow=8, figsize=(10,10)):
    _, _= plt.subplots(1,1, figsize=figsize)
    sample = sample.long() [:num_img]
    img = make_grid(sample, nrow=nrow)
    npimg = np.transpose(img.detach().cpu().numpy(),(1,2,0))
    plt.imshow(npimg)
    plt.axis('off')
    plt.savefig(save_path+'/selected_sample.png')
    plt.show()



if __name__ == "__main__":

    import argparse
    import os
    import torch

    if torch.cuda.is_available():

        #..Parse the arguments

        parser = argparse.ArgumentParser(description='Run image generation.')
        parser.add_argument('--path', type=str, required=True, help='path to model')
        parser.add_argument('--sample_size', type=int, required=False, help='number of samples', default=100)
        parser.add_argument('--num_timesteps', type=int, required=False, help='number of time steps', default=100)
        parser.add_argument('--timepsilon', type=float, required=False, help='time epsilon', default=0.0)

        arg = parser.parse_args()

        path = os.path.join(results_path, "crm", arg.path)

        source = get_cifar10_test_samples(path, which='source', sample_size=arg.sample_size, device="cuda:0")
        gen_samples, _ , _ = generate_samples(path, source, num_timesteps=arg.num_timesteps, time_epsilon=arg.timepsilon, device="cuda:0")
        image_grid(gen_samples, num_img=arg.sample_size, nrow=int(np.sqrt(arg.sample_size)), figsize=(3,3), save_path=path)
        cifar_noise_bridge(path, source[:6],  num_timesteps=arg.num_timesteps, time_epsilon=arg.timepsilon, num_img=6, num_timesteps_displayed=20, save_path=path)