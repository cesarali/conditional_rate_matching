import os
import torch
import numpy as np

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.generative_models.ctdd import CTDD
from conditional_rate_matching import results_path
from conditional_rate_matching.models.metrics.fid_metrics import load_classifier
from conditional_rate_matching.models.metrics.fid_metrics import fid_nist
from conditional_rate_matching.models.metrics.distances import marginal_histograms, kmmd 

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def run_analysis(path,
                 run="run",
                 generative_model = "crm",
                 num_timesteps = 100,
                 time_epsilon = 0.01,
                 num_img_bridge = 6, 
                 num_intermediate_bridge = 20,
                 device = "cpu"):
    
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

    mnist_grid(x_1[:100], save_path=experiment_dir, num_img=100, nrow=10, figsize=(4, 4))
    mnist_classifier(x_1, save_path=experiment_dir, plot_histogram=True)
    get_fid(x_1, x_test, experiment_dir)
    get_nist_metrics(x_1, x_test, experiment_dir)

    if not os.path.isfile(experiment_dir + "/bridge_example.png") and generative_model == "crm":
        mnist_noise_bridge(experiment_dir,
                        x_0, 
                        num_timesteps=num_timesteps,  
                        time_epsilon=time_epsilon,
                        num_img=num_img_bridge, 
                        num_timesteps_displayed=num_intermediate_bridge, 
                        save_path=experiment_dir) 


def get_mnist_test_samples(trained_model, 
                           class_label=None, 
                           sample_size=800, 
                           device="cpu"):
    
    experiment_dir = os.path.join(results_path,"crm", trained_model)
    crm = CRM(experiment_dir=experiment_dir, device=device)
    images = []
    for batch in crm.dataloader_0.test():
        if len(batch) == 2:
            sample, labels = batch[0], batch[1]
            selected_images = sample[labels == class_label] if class_label is not None else sample 
            images.append(selected_images)
        else:
            sample = batch[0].view(-1, 1, 28, 28)
            images.append(sample)

    return torch.cat(images, dim=0)[:sample_size].to(device) #if labeled else torch.tensor(images, device=device)

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
    
    x_1 = x_1.view(-1, 1, 28, 28)
    x_t = x_t.view(-1, x_t.shape[1], 1, 28, 28)
    return x_1, x_t, t

def generate_mnist_samples(path,
                           generative_model='crm', 
                           num_timesteps=100,
                           time_epsilon=0.005, 
                           class_label=None, 
                           device="cpu"):
    
    if generative_model == 'ctdd':

        ctdd = CTDD(experiment_dir=path, device=device)
        target = ctdd.dataloader_1.test()
     
        x_0, x_1, x_test = [], [], []

        for batch in target:
            if len(batch) == 2: test_images, labels = batch[0], batch[1]
            else: test_images = batch[0]
            x_test.append(test_images)
        
        for _ in range(10):
            input_images = ctdd.pipeline.get_x0_sample(train=False, sample_size=1000)
            gen_images = ctdd.pipeline(ctdd.backward_rate, train=False, sample_size=1000) 
            x_0.append(input_images)
            x_1.append(gen_images.detach().cpu())

        x_test = torch.cat(x_test).view(-1, 1, 28, 28)
        x_0 = torch.cat(x_0).view(-1, 1, 28, 28)
        x_1 = torch.cat(x_1).view(-1, 1, 28, 28)

    if generative_model == 'crm':

        crm = CRM(experiment_dir=path, device=device)
        crm.config.pipeline.time_epsilon = time_epsilon
        crm.config.pipeline.num_intermediates = num_timesteps
        crm.config.pipeline.number_of_steps = num_timesteps
        source = crm.dataloader_0.test()
        target = crm.dataloader_1.test()

        x_1, x_0, x_test = [], [], []

        for batch in target:
            if len(batch) == 2: test_images, labels = batch[0], batch[1]
            else: test_images = batch[0]
            x_test.append(test_images)
        
        for batch in source:
            if len(batch) == 2:
                sample, labels = batch[0], batch[1]
                input_images = sample[labels == class_label] if class_label is not None else sample 
            else:
                input_images = batch[0]
                
            gen_images = crm.pipeline(sample_size=input_images.shape[0], return_intermediaries=False, train=False, x_0=input_images.to(crm.device))
            x_0.append(input_images)
            x_1.append(gen_images.detach().cpu())

        x_test = torch.cat(x_test)
        x_0 = torch.cat(x_0, dim=0).view(-1, 1, 28, 28)
        x_1 = torch.cat(x_1, dim=0).view(-1, 1, 28, 28)

    torch.save(x_0, os.path.join(path, "sample_gen_x0.dat"))      
    torch.save(x_1, os.path.join(path, "sample_gen_x1.dat"))      
    torch.save(x_test, os.path.join(path, "sample_gen_test.dat"))
    
    return x_0, x_1, x_test

def mnist_classifier(img, save_path=None, plot_histogram=False):
    device = img.device
    classifier = load_classifier('mnist', device=device)
    classifier = classifier.to(device)
    classifier.eval()
    y = classifier(img)
    classes = torch.argmax(y, dim=1).cpu().numpy()
    classes = classes.tolist()
    if plot_histogram:
        plt.subplots(figsize=(3,3))
        unique, counts = np.unique(classes, return_counts=True)
        plt.bar(unique, counts, color='darkred')
        plt.xticks(range(10))
        plt.xlabel('digit ocurrence')
        plt.tight_layout()
        plt.savefig(save_path+'/class_ocurrence.png')
        plt.show()
    else:
        return torch.Tensor(classes)
    
def mnist_grid(sample, save_path='.', num_img=5, nrow=8, figsize=(10,10)):
    _, _= plt.subplots(1,1, figsize=figsize)
    sample = sample[:num_img]
    img = make_grid(sample, nrow=nrow)
    npimg = np.transpose(img.detach().cpu().numpy(),(1,2,0))
    plt.imshow(npimg)
    plt.axis('off')
    plt.savefig(save_path+'/selected_sample.png')
    plt.show()

def mnist_noise_bridge(path, 
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
    
    for j, idx in enumerate(np.arange(0, N+1, dt)):
        if j<num_timesteps_displayed:
            tau = time_steps[idx]
            images = img_hist[:, idx, :]
        else:
            tau = time_steps[-1]
            images = img_hist[:, -1, :]
            
        for i in range(num_img):
            img = images[i].detach().cpu().numpy()
            axs[i, j].imshow(img.squeeze(), cmap='gray')
            if i == 0: axs[i, j].set_title(r'$\tau = {0}$'.format(round(tau.item(),2)))
            axs[i, j].axis('off')
    
    j = num_timesteps_displayed + 1
    for i in range(num_img):
        axs[i, j].imshow(img_1[i].detach().cpu().numpy().squeeze(), cmap='gray')
        axs[i, j].axis('off')

    plt.tight_layout()
    if save_path is None: plt.show()
    else: plt.savefig(save_path+'/bridge_example.png')


def get_fid(x_1, x_test, save_path=None):
    fids = fid_nist(x_1, x_test, 'mnist', x_1.device)
    if save_path:
        fid_path = os.path.join(save_path, "fid.txt")
        with open(fid_path, 'w') as f:
            f.write(str(fids))
    return fids

def get_nist_metrics(x_1, x_test, save_path=None):
    metrics = fid_nist(x_1, x_test, 'mnist', x_1.device)
    metrics["mse"] = marginal_histograms(x_1, x_test)
    print(metrics)
    if save_path:
        metrics_path = os.path.join(save_path, "metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(str(metrics))
    return metrics

# def get_fid(x_1, x_test, save_path=None):
#     fid = fid_nist(x_1, x_test, 'mnist', x_1.device)
#     fid_avg = 0
#     for f in fid.values():
#         fid_avg+=f/3.0
#     fid['fid_avg'] = fid_avg
#     return fid