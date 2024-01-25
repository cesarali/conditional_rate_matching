import os
import json
import joblib
from urllib.request import urlretrieve

from conditional_rate_matching.data.dataloaders_utils import get_data
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist

from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Main file which contrains all DDSM logic
from conditional_rate_matching.models.generative_models.ddsm import *
from conditional_rate_matching import data_path
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from functools import reduce
from conditional_rate_matching.configs.config_files import ExperimentFiles
from torch.distributions import Bernoulli
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader
from conditional_rate_matching.models.metrics.fid_metrics import fid_nist

#python presample_noise.py -n  -c 2 -t 4000 --max_time 4 --out_path binary_mnist/

max_test_size = 1000 # 4000
DEBUG = False
device = "cuda:3" 

#=================================
# PRESAMPLED NOISE VARIABLE
#=================================

noise_dir = os.path.join(data_path,"raw")

if DEBUG:
    num_time_steps = 40
    num_samples = 100
else:
    num_time_steps = 4000
    num_samples = 100000

num_cat = 2
str_speed = ""
max_time = 4.0
C = 2
num_epochs = 50
lr = 5e-4


out_path = noise_dir

filename = f'steps{num_time_steps}.cat{num_cat}{str_speed}.time{max_time}.' \
           f'samples{num_samples}'
filepath = os.path.join(out_path, filename + ".pth")

# We have two categories, so the beta parameters are
alpha = torch.DoubleTensor([1.0])
beta = torch.DoubleTensor([1.0])

def binary_to_onehot(x):
    xonehot = []
    xonehot.append((x == 1)[..., None])
    xonehot.append((x == 0)[..., None])
    return torch.cat(xonehot, -1)

if not os.path.exists(filepath):
    
    print("Generating Noise")

    alpha = torch.ones(num_cat - 1).to(device)
    beta = torch.arange(num_cat - 1, 0, -1).to(device)
    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = noise_factory(num_samples,
                                                                                num_time_steps,
                                                                                alpha,
                                                                                beta,
                                                                                total_time=max_time,
                                                                                order=1000,
                                                                                time_steps=200,
                                                                                logspace=None,
                                                                                speed_balanced=False,
                                                                                mode="path",
                                                                                device=device)
    v_one = v_one.cpu().float()
    v_zero = v_zero.cpu().float()
    v_one_loggrad = v_one_loggrad.cpu().float()
    v_zero_loggrad = v_zero_loggrad.cpu().float()
    timepoints = torch.FloatTensor(timepoints)
    torch.save((v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints), filepath)
else:
    v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load(filepath)

experiment_files = ExperimentFiles(experiment_name="ddsm", experiment_type="test")
experiment_files.create_directories()
configs = experiment_nist(temporal_network_name="unet_conv")
configs.data1 = NISTLoaderConfig(flatten= False,as_image=True)
training_dl, test_dl = get_data(configs.data1)
weights_est_dl = training_dl

sample_size = 1000
remaining = sample_size
valid_datasets = []

for x, _ in test_dl:
    batch_size = x.size(0)
    take_num = min(batch_size, remaining)
    x = x[:take_num]
    valid_datasets.append(x)
    remaining -= take_num
    if remaining <= 0:
        break
#==================================
# model
# ==================================

score_model = load_temporal_network(configs,torch.device(device))
score_model.to(device)
score_model.train()
speed_balanced = False

if speed_balanced:
    s = 2 / (
            torch.ones(C - 1, device=device)
            + torch.arange(C - 1, 0, -1, device=device).float()
    )
else:
    s = torch.ones(C - 1, device=device)

# Number of epochs for training
    
sb = UnitStickBreakingTransform()
n_time_steps = timepoints.shape[0]
time_dependent_cums = torch.zeros(n_time_steps).to(device)
time_dependent_counts = torch.zeros(n_time_steps).to(device)

for i, x in enumerate(weights_est_dl):
    x = x[0]
    x = binary_to_onehot(x.squeeze())
    random_t = torch.randint(0, n_time_steps, (x.shape[0],))
    order = np.random.permutation(np.arange(C))
    perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device=device)
    perturbed_x = perturbed_x.to(device)
    perturbed_x_grad = perturbed_x_grad.to(device)
    random_t = random_t.to(device)

    # Transform data from x space to v space, where diffusion happens
    order = np.random.permutation(np.arange(C))
    perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()

    # Computing score. gx_to_gv transforms gradient in x space to gradient in v space
    time_dependent_cums[random_t] += (
        (perturbed_v * (1 - perturbed_v)
            * s[(None,) * (x.ndim - 1)]
            * gx_to_gv(perturbed_x_grad, perturbed_x, compute_gradlogdet=False) ** 2
            ).view(x.shape[0], -1).mean(dim=1).detach()
    )

    time_dependent_counts[random_t] += 1

time_dependent_weights = time_dependent_cums / time_dependent_counts
time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()

#the score loss is computed in the v-space
def loss_fn(score, perturbed_x_grad, perturbed_x, important_sampling_weights=None):
    perturbed_v = sb._inverse(perturbed_x, prevent_nan=True).detach()
    if important_sampling_weights is not None:
        important_sampling_weights = 1 / important_sampling_weights[(...,) + (None,) * (x.ndim - 1)]
    else:
        important_sampling_weights = 1
    loss = torch.mean(
            torch.mean(
                important_sampling_weights
                * s[(None,) * (x.ndim - 1)]
                * perturbed_v * (1 - perturbed_v)
                * (gx_to_gv(score, perturbed_x, create_graph=True, compute_gradlogdet=False) - gx_to_gv(perturbed_x_grad, perturbed_x, compute_gradlogdet=False)) ** 2,
                dim=(1))
        )
    return loss

torch.set_default_dtype(torch.float32)

# Defining optimizer
optimizer = Adam(score_model.parameters(), lr=lr, weight_decay=1e-10)
timepoints = timepoints.to(device)
tqdm_epoch = tqdm.tqdm(range(num_epochs))  

for epoch in tqdm_epoch:
    avg_loss = 0.0
    num_items = 0

    for x in training_dl:
        x = x[0]
        x = binary_to_onehot(x.squeeze())
        random_t = torch.LongTensor(
            np.random.choice(
                np.arange(n_time_steps),
                size=x.shape[0],
                p=(
                        torch.sqrt(time_dependent_weights)
                        / torch.sqrt(time_dependent_weights).sum()
                )
                .cpu()
                .detach()
                .numpy(),
            )
        ).to(device)

        # Similarly to computing importance sampling weights, there are two options
        perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device=device)
        perturbed_x = perturbed_x.to(device)
        perturbed_x_grad = perturbed_x_grad.to(device)
        random_t = random_t.to(device)
        random_timepoints = timepoints[random_t]

        # Doing score estimation via neural network
        score = score_model(perturbed_x, random_timepoints)

        # Computing loss function
        loss = loss_fn(score, 
                       perturbed_x_grad, 
                       perturbed_x,
                       important_sampling_weights=(torch.sqrt(time_dependent_weights))[random_t])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    # Print the averaged training loss so far.
    tqdm_epoch.set_description("Average Loss: {:5f}".format(avg_loss / num_items))

    # Doing validation check every 10 epochs.
    if epoch % 10 == 0:
        valid_avg_loss = 0.0
        valid_num_items = 0

        for x in valid_datasets:
            x = binary_to_onehot(x.squeeze())
            random_t = torch.LongTensor(
                np.random.choice(
                    np.arange(n_time_steps),
                    size=x.shape[0],
                    p=(
                            torch.sqrt(time_dependent_weights)
                            / torch.sqrt(time_dependent_weights).sum()
                    ).cpu().detach().numpy(),
                )
            ).to(device)

            perturbed_x, perturbed_x_grad = diffusion_factory(x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device=device)
            perturbed_x = perturbed_x.to(device)
            perturbed_x_grad = perturbed_x_grad.to(device)
            random_t = random_t.to(device)
            random_timepoints = timepoints[random_t]
            score = score_model(perturbed_x, random_timepoints)
            loss = loss_fn(score, 
                           perturbed_x_grad, 
                           perturbed_x, 
                           important_sampling_weights=(torch.sqrt(time_dependent_weights))[random_t])

            valid_avg_loss += loss.item() * x.shape[0]
            valid_num_items += x.shape[0]

        print("Average Loss: {:5f}".format(valid_avg_loss / valid_num_items))
    if DEBUG:
        break

torch.save({"model":score_model}, experiment_files.best_model_path)

#====================================================
# SAMPLES
#====================================================
test_sample = sample_from_dataloader(training_dl, sample_size=max_test_size).to(torch.device(device))
sampler = Euler_Maruyama_sampler  ## Generate samples using the specified sampler.
samples = sampler(score_model, (28, 28, 2),
                    batch_size=max_test_size,
                    max_time=4,
                    min_time=0.01,
                    num_steps=100,
                    eps=1e-5,
                    device=device)

samples = samples.clamp(0.0, 1.0)

pixel_probability = samples[:, :, :, 0]
pixel_distribution = Bernoulli(pixel_probability)
generative_sample = pixel_distribution.sample().unsqueeze(1)
mse_metric_path = experiment_files.metrics_file.format("fid_nist" + "_{0}_".format("best"))
fid_nist_metrics = fid_nist(generative_sample,test_sample,dataset_name="mnist",device=device)

with open(mse_metric_path, "w") as f:
    json.dump(fid_nist_metrics, f)
