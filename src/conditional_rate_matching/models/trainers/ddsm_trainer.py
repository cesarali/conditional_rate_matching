import os
import json
import joblib
from urllib.request import urlretrieve

from conditional_rate_matching.data.dataloaders_utils import get_data
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network

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

max_test_size = 1000 #4000
DEBUG = False
device = "cuda:1"  # alternative option is "cpu"

#=================================
# PRESAMPLED NOISE VARIABLE
#=================================

noise_dir = os.path.join(data_path,"raw")
new_dl = True

if DEBUG:
    num_time_steps = 40
    num_samples = 100
else:
    num_time_steps = 4000
    num_samples = 100000

num_cat = 2
str_speed = ""
max_time = 4.0

out_path = noise_dir

# number of categories are 2 for binarized MNIST data
C = 2
num_epochs = 50
lr = 5e-4

# device, where code will be run

# setting speed_balanced does not have an effect for C=2
# For >2 categories, setting speed_balanced flag affects the convergence speed of the forward diffusion.
# (see Appendix A.2)
# speed_balanced = True leads to similar convergence speed across individual univariate Jacobi diffusion processes
# speed balanced = False leads to similar convergence speed across different categories after stick breaking transform

filename = f'steps{num_time_steps}.cat{num_cat}{str_speed}.time{max_time}.' \
           f'samples{num_samples}'
filepath = os.path.join(out_path, filename + ".pth")

# We have two categories, so the beta parameters are
alpha = torch.DoubleTensor([1.0])
beta = torch.DoubleTensor([1.0])

# Example of 4 categories to be used with diffusion_factory
# alpha = torch.DoubleTensor([1.0, 1.0, 1.0])
# beta = torch.DoubleTensor([3.0, 2.0, 1.0])

# Model architecture borrowed from https://github.com/yang-song/score_sde
# Note the manuscript experiment on binarized MNIST use a different, larger model

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]

class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(2, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )

        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )

        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 2, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t / 4.0))
        # Encoding path
        h1 = self.conv1(x.permute(0, 3, 1, 2))
        ## Incorporate information from t
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # some scaling here by time or x may be helpful but no scaling works fine here
        h = h.permute(0, 2, 3, 1)
        h = h - h.mean(axis=-1, keepdims=True)
        return h

class ScoreNetMLP(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.config = config
        self.vocab_size = config.data1.vocab_size
        self.dimensions = config.data1.dimensions

        self.f1 = nn.Linear(28*28*2,self.dimensions)
        self.expected_data_shape = config.data1.temporal_net_expected_shape
        self.define_deep_models(config,device)

    def forward(self,x,t):
        batch_size = x.size(0)
        x = x.view(batch_size,-1)
        h = self.f1(x)
        h = self.temporal_network(h,t)
        #h = self.temporal_to_rate(h)
        h = h.view(-1,28,28,2)
        return h

    def define_deep_models(self, config, device):
        self.temporal_network = load_temporal_network(config, device=device)
        self.expected_temporal_output_shape = self.temporal_network.expected_output_shape
        if self.expected_temporal_output_shape != [self.dimensions, self.vocab_size]:
            temporal_output_total = reduce(lambda x, y: x * y, self.expected_temporal_output_shape)
            self.temporal_to_rate = nn.Linear(temporal_output_total, self.dimensions * self.vocab_size)

#============================================================================
# DATA
#============================================================================

def load_mnist_binarized(root):
    datapath = os.path.join(root, "bin-mnist")
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    dataset = os.path.join(datapath, "mnist.pkl.gz")

    if not os.path.isfile(dataset):

        datafiles = {
            "train": "http://www.cs.toronto.edu/~larocheh/public/"
            "datasets/binarized_mnist/binarized_mnist_train.amat",
            "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
            "binarized_mnist/binarized_mnist_valid.amat",
            "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
            "binarized_mnist/binarized_mnist_test.amat",
        }
        datasplits = {}
        for split in datafiles.keys():
            print("Downloading %s data..." % (split))
            datasplits[split] = np.loadtxt(urlretrieve(datafiles[split])[0])

        joblib.dump(
            [datasplits["train"], datasplits["valid"], datasplits["test"]],
            open(dataset, "wb"),
        )

    x_train, x_valid, x_test = joblib.load(open(dataset, "rb"))
    return x_train, x_valid, x_test

class BinMNIST(Dataset):
    """Binary MNIST dataset"""

    def __init__(self, data, device="cpu", transform=None):
        h, w, c = 28, 28, 1
        self.device = device
        self.data = torch.tensor(data, dtype=torch.float).view(-1, c, h, w)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample.to(self.device)

def get_binmnist_datasets(root, device="cpu"):
    x_train, x_valid, x_test = load_mnist_binarized(root)
    # x_train = np.append(x_train, x_valid, axis=0)  # https://github.com/casperkaae/LVAE/blob/master/run_models.py (line 401)
    return (
        BinMNIST(x_train, device=device),
        BinMNIST(x_valid, device=device),
        BinMNIST(x_test, device=device),
    )

def binary_to_onehot(x):
    xonehot = []
    xonehot.append((x == 1)[..., None])
    xonehot.append((x == 0)[..., None])
    return torch.cat(xonehot, -1)

if __name__=="__main__":
    if not os.path.exists(filepath):
        print("Generating Noise")

        #torch.set_default_dtype(torch.float64)

        device = "cuda:1" # "cuda:1"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if new_dl:
        from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist

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

        #configs.temporal_network = TemporalMLPConfig()
        #configs.temporal_network = TemporalDeepMLPConfig(num_layers=1)
        #score_model = ScoreNetMLP(configs)

        score_model = load_temporal_network(configs,torch.device(device))
        score_model.to(device)
        score_model.train()

    else:
        train_set, valid_set, test_set = get_binmnist_datasets("./mnist")
        batch_size = 32
        weights_est_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)#, num_workers=4)
        training_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)#, num_workers=4)
        valid_data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
        valid_datasets = []
        for x in valid_data_loader:
            valid_datasets.append(x)
        test_dl = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

        # Defining a model
        score_model = ScoreNet(channels=[64, 128, 256, 512], embed_dim=512)
        score_model = score_model.to(device)
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
        if new_dl:
            x = x[0]
        x = binary_to_onehot(x.squeeze())
        random_t = torch.randint(0, n_time_steps, (x.shape[0],))
        order = np.random.permutation(np.arange(C))

        # There are two options for using presampled noise:
        # First one is regular approach and second one is fast sampling (see Appendix A.4 for more info)
        perturbed_x, perturbed_x_grad = diffusion_factory(
            x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device=device,
        )
        # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)

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
            important_sampling_weights = 1/important_sampling_weights[
                        (...,) + (None,) * (x.ndim - 1)]
        else:
            important_sampling_weights = 1
        loss = torch.mean(
                torch.mean(
                    important_sampling_weights
                    * s[(None,) * (x.ndim - 1)]
                    * perturbed_v * (1 - perturbed_v)
                    * (gx_to_gv(
                            score, perturbed_x, create_graph=True, compute_gradlogdet=False
                        ) - gx_to_gv(perturbed_x_grad, perturbed_x, compute_gradlogdet=False)
                      ) ** 2,
                    dim=(1))
            )
        return loss

    torch.set_default_dtype(torch.float32)

    # Defining optimizer
    optimizer = Adam(score_model.parameters(), lr=lr, weight_decay=1e-10)
    timepoints = timepoints.to(device) #############################<-------------------HERE

    # tqdm_epoch = tqdm.notebook.trange(num_epochs)
    tqdm_epoch = tqdm.tqdm(range(num_epochs))   
    for epoch in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0

        for x in training_dl:
            if new_dl:
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
            perturbed_x, perturbed_x_grad = diffusion_factory(
                x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device=device,
            )
            # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)

            perturbed_x = perturbed_x.to(device)
            perturbed_x_grad = perturbed_x_grad.to(device)
            random_t = random_t.to(device)
            random_timepoints = timepoints[random_t]

            # Doing score estimation via neural network
            score = score_model(perturbed_x, random_timepoints)

            # Computing loss function
            loss = loss_fn(score, perturbed_x_grad, perturbed_x,
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

                perturbed_x, perturbed_x_grad = diffusion_factory(
                    x, random_t, v_one, v_zero, v_one_loggrad, v_zero_loggrad, alpha, beta, device=device,
                )
                # perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x, random_t, v_one, v_one_loggrad)

                perturbed_x = perturbed_x.to(device)
                perturbed_x_grad = perturbed_x_grad.to(device)
                random_t = random_t.to(device)
                random_timepoints = timepoints[random_t]

                score = score_model(perturbed_x, random_timepoints)
                loss = loss_fn(score, perturbed_x_grad, perturbed_x,
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
