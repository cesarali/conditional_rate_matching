import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def frechet_distance(mu_1, sigma_1, mu_2, sigma_2):
    '''
    Returns:
    - fid:  The Frechet distance between two Gaussians d = ||mu_1 - mu_2||^2 + Trace(sig_1 + sig_2 - 2*sqrt(sig_1 * sig_2)).
    '''
    mse = (mu_1 - mu_2).square().sum(dim=-1)
    trace = sigma_1.trace() + sigma_2.trace() - 2 * torch.linalg.eigvals(sigma_1 @ sigma_2).sqrt().real.sum(dim=-1)
    return mse + trace

@torch.no_grad()
def compute_activation_statistics(classifier, data, activation_layer='fc1', device='cpu'):
    '''
     Args:

     - classifier       : Instance of pre-trained classifier network
     - data             : The input data as a tensor
     - activation_layer : select the activation layer of the classifier network, e.g. 'fc1', 'fc2' or 'fc3' for LeNet-5

     Returns:

     - mu    : The sample mean of the activations of the activation_layer of the classifier.
     - sigma : The sample covariance of the activations of the activation_layer of the classifier.

    '''   
    activations = classifier(data.to(device), activation_layer=activation_layer)
    if len(activations.shape) > 2:
        activations = F.adaptive_avg_pool2d(activations, (1, 1)).view(activations.size(0), -1)
    mu = torch.mean(activations, dim=0)
    sigma = torch.cov(activations.t())
    return mu, sigma


def compute_fid(name, classifier, gen_sample, test_sample, activation_layer='fc1', device='cpu'):
    
    if name in ['mnist', 'emnist', 'fashion_mnist']:
        assert gen_sample.shape[1] == 784, " data should have a flatten mnist with shape (N, 784)"
        gen_sample = gen_sample.reshape(gen_sample.shape[0], 1, 28, 28)
        test_sample = test_sample.reshape(test_sample.shape[0], 1, 28, 28)

    classifier.to(device)
    classifier.eval()  
    mu_test, sigma_test = compute_activation_statistics(classifier, test_sample, activation_layer, device)
    mu_gen, sigma_gen = compute_activation_statistics(classifier, gen_sample, activation_layer, device)
    fid = frechet_distance(mu_gen, sigma_gen, mu_test, sigma_test)
    return fid