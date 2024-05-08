import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from conditional_rate_matching.models.metrics.fid_nist.image_datasets import load_nist_data

def frechet_distance(mu_1, sigma_1, mu_2, sigma_2):
    '''
    Returns:
    - fid:  The Frechet distance between two Gaussians d = ||mu_1 - mu_2||^2 + Trace(sig_1 + sig_2 - 2*sqrt(sig_1 * sig_2)).
    '''
    mse = (mu_1 - mu_2).square().sum(dim=-1)
    trace = sigma_1.trace() + sigma_2.trace() - 2 * torch.linalg.eigvals(sigma_1 @ sigma_2).sqrt().real.sum(dim=-1)
    return mse + trace

@torch.no_grad()
def compute_activation_statistics(model, dataset, batch_size=64, activation_layer='fc1', device='cpu'):
    model.to(device)
    model.eval()
    features = []

    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        if len(batch) > 1:
            batch,_ = batch
            batch = batch.view(-1,1,28,28)
        else:
            batch: torch.Tensor
            batch = batch[0]
            batch = batch.view(-1,1,28,28)
        batch = batch.to(device)
        activations = model(batch, activation_layer=activation_layer)

        #...apply global average pooling if features are not 2D

        if len(activations.shape) > 2:
            activations = F.adaptive_avg_pool2d(activations, (1, 1)).view(activations.size(0), -1)

        features.append(activations)

    features = torch.cat(features, dim=0)
    mu = torch.mean(features, dim=0)
    sigma = torch.cov(features.t())
    return mu, sigma

def compute_fid(model, dataset, dataset_ref=None, mu_ref=None, sigma_ref=None, batch_size=64, activation_layer='fc1', device='cpu'):
    
    assert dataset_ref is not None or (mu_ref is not None and sigma_ref is not None), 'Either dataset_ref or (mu_ref, sigma_ref) must be provided.'
 
    if dataset_ref is None:
        mu, sigma = compute_activation_statistics(model, dataset, batch_size, activation_layer, device)
    else:
        mu_ref, sigma_ref = compute_activation_statistics(model, dataset_ref, batch_size, activation_layer, device)
        mu, sigma = compute_activation_statistics(model, dataset, batch_size, activation_layer, device)

    return frechet_distance(mu, sigma, mu_ref, sigma_ref)


def fid_distorted_NIST(model, name='MNIST', distortion='noise', values=np.arange(0.0, 1, 0.02), batch_size=64, activation_layer='fc1', device='cpu'):
    
    dataset = load_nist_data(name=name, train=False)
    mu, sigma = compute_activation_statistics(model, dataset, batch_size=batch_size, activation_layer=activation_layer, device=device)
    fid = {}
    
    for val in values:
        dataset = load_nist_data(name, distortion=distortion, level=val, train=False)
        fid[val] = compute_fid(model, dataset, mu_ref=mu, sigma_ref=sigma, batch_size=batch_size, activation_layer=activation_layer, device=device).cpu()
    return fid


if __name__=="__main__":
    import torch
    from architectures import LeNet5

    device = 'cpu'
    model = LeNet5(num_classes=10)
    model.load_state_dict(torch.load('./models/LeNet5_BinaryMNIST.pth'))
    model.eval()

    # ...Load MNIST test dataset
    test_ref = load_nist_data(name='MNIST', train=False)

    # ...compute mean and std of features from reference data for each layer:
    mu_1, sigma_1 = compute_activation_statistics(model, test_ref, activation_layer='fc1', device=device)
    mu_2, sigma_2 = compute_activation_statistics(model, test_ref, activation_layer='fc2', device=device)
    mu_3, sigma_3 = compute_activation_statistics(model, test_ref, activation_layer='fc3', device=device)

    data = load_nist_data(name='MNIST', train=False, distortion='noise', level=0.25)
    fid_1 = compute_fid(model, data, mu_ref=mu_1, sigma_ref=sigma_1, activation_layer='fc1', device=device)
    fid_2 = compute_fid(model, data, mu_ref=mu_2, sigma_ref=sigma_2, activation_layer='fc2', device=device)
    fid_3 = compute_fid(model, data, mu_ref=mu_3, sigma_ref=sigma_3, activation_layer='fc3', device=device)
    print(fid_1)