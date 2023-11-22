import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import Optional,Union,Tuple
from conditional_rate_matching.configs.config_oops import OopsConfig

import torch.distributions as dists
from conditional_rate_matching.data.image_dataloaders import NISTLoader
from conditional_rate_matching.utils.devices import check_model_devices
from conditional_rate_matching.models.temporal_networks.ebm import EBM
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles

from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.data.image_dataloaders import NISTLoaderConfig
from conditional_rate_matching.models.pipelines.mc_samplers.oops_samplers import PerDimGibbsSampler,DiffSampler


from torch import nn
from conditional_rate_matching.utils.devices import check_model_devices

def preprocess(data,config:OopsConfig):
    if config.pipeline.dynamic_binarization:
        return torch.bernoulli(data)
    else:
        return data

class AISModel(nn.Module):
    def __init__(self, model, init_dist):
        super().__init__()
        self.model = model
        self.init_dist = init_dist

    def forward(self, x, beta):
        logpx = self.model(x).squeeze()
        logpi = self.init_dist.log_prob(x).sum(-1)
        return logpx * beta + logpi * (1. - beta)

class OopsPipeline:
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    config : OopsConfig
    model: EBM
    data: NISTLoader

    def __init__(self,
                 config:OopsConfig,
                 sampler:Union[PerDimGibbsSampler,DiffSampler],
                 dataloader:NISTLoaderConfig,
                 experiment_files:ExperimentFiles=None):

        super().__init__()
        self.oops_config = config

        # register models
        self.dataloader = dataloader
        self.sampler = sampler
        self.experiment_files = experiment_files

        #parameters
        self.n_iters = self.oops_config.pipeline.n_iters
        self.n_samples = self.oops_config.pipeline.n_samples
        self.buffer_init = self.oops_config.pipeline.buffer_init
        self.buffer_size = self.oops_config.pipeline.buffer_size
        self.steps_per_iter = self.oops_config.pipeline.steps_per_iter
        self.viz_every = self.oops_config.pipeline.viz_every
        self.input_type = self.oops_config.pipeline.input_type

    @torch.no_grad()
    def __call__(
        self,
        model: Optional[EBM] = None,
        sample_size = None,
    ) -> Union[torch.Tensor, Tuple]:
        """

        :param model:
        :param sinkhorn_iteration:
        :param device:
        :return:
        """
        device = check_model_devices(model)
        sample_size = sample_size or self.oops_config.pipeline.num_samples
        x = self.evaluate(model,device)
        return x

    def evaluate(self,model,device):
        ais_model = AISModel(model, self.init_dist)

        # move to cuda
        ais_model.to(device)

        # annealing weights
        betas = np.linspace(0., 1., self.n_iters)

        samples = self.init_dist.sample((self.n_samples,)).to(device)
        log_w = torch.zeros((self.n_samples,)).to(device)

        gen_samples = []
        for itr, beta_k in tqdm(enumerate(betas)):
            if itr == 0:
                continue  # skip 0

            beta_km1 = betas[itr - 1]

            # udpate importance weights
            with torch.no_grad():
                log_w = log_w + ais_model(samples, beta_k) - ais_model(samples, beta_km1)
            # update samples
            model_k = lambda x: ais_model(x, beta=beta_k)
            for d in range(self.steps_per_iter):
                samples = self.sampler.step(samples.detach(), model_k).detach()

            if (itr + 1) % self.viz_every == 0:
                gen_samples.append(samples.cpu().detach())

        #ll = self.ll(ais_model.model,log_w,device)

        return gen_samples

    def ll(self,model,log_w,device):
        logZ_final = log_w.logsumexp(0) - np.log(self.n_samples)
        print("Final log(Z) = {:.4f}".format(logZ_final))

        logps = []
        for x, _ in self.dataloader.train():
            x = preprocess(x.to(device))
            logp_x = model(x).squeeze().detach()
            logps.append(logp_x)

        logps = torch.cat(logps)
        ll = logps.mean() - logZ_final
        return ll

    def get_buffer(self,device):
        data_path = Path(self.oops_config.data0.data_dir) / self.oops_config.data0.dataset_name
        buffer_path = data_path / f"buffer_{self.buffer_init}.pkl"

        if buffer_path.exists():
            # Load the buffer if it exists
            self.buffer = torch.load(buffer_path)
            init_mean = self.get_init_mean(False)
        else:
            # Create the buffer based on specified initialization
            if self.buffer_init == "mean":
                init_mean = self.get_init_mean(False)
                if self.input_type == "binary":
                    init_dist = torch.distributions.Bernoulli(probs=init_mean)
                    self.buffer = init_dist.sample((self.buffer_size,))
                else:
                    self.buffer = None
                    raise ValueError("Other types of data not yet implemented")

            elif self.buffer_init == "data":
                init_mean,init_batch = self.get_init_mean(True)
                all_inds = list(range(init_batch.size(0)))
                init_inds = np.random.choice(all_inds, self.buffer_size)
                self.buffer = init_batch[init_inds]
            elif self.buffer_init == "uniform":
                init_mean, init_batch = self.get_init_mean(True)
                self.buffer = (torch.ones(self.buffer_size, *init_batch.size()[1:]) * .5).bernoulli()
            else:
                raise ValueError("Invalid init")

            # Save the newly created buffer
            torch.save(self.buffer, buffer_path)

        self.init_dist = torch.distributions.Bernoulli(probs=init_mean.to(device))
        self.reinit_dist = torch.distributions.Bernoulli(probs=torch.tensor(self.oops_config.pipeline.reinit_freq))

    def get_init_mean(self,return_init_batch=False):
        data_path = Path(self.oops_config.data0.data_dir)
        init_mean_path = data_path / "init_mean.pkl"
        if not init_mean_path.exists():
            init_batch = []
            for databatch in self.dataloader.train():
                x = databatch[0]
                init_batch.append(x)
                break
            init_batch = torch.cat(init_batch, 0)
            eps = 1e-2
            init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps
            pickle.dump(init_mean,open(init_mean_path,"wb"))
        else:
            init_mean = pickle.load(open(init_mean_path,"rb"))

        if return_init_batch:
            return init_mean,init_batch
        else:
            return init_mean

