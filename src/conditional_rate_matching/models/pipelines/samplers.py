import torch
import numpy as np
from tqdm import tqdm
from torch import functional as F
from typing import Union,Tuple,List

from conditional_rate_matching.configs.config_crm import Config as ConditionalRateMatchingConfig
from conditional_rate_matching.models.generative_models.crm import ConditionalBackwardRate


def TauLeaping(config:ConditionalRateMatchingConfig,
               rate_model:Union[ConditionalBackwardRate],
               x_0:torch.Tensor,
               forward=True):
    """
    :param rate_model:
    :param x_0:
    :param N:
    :param num_intermediates:
    :return:
    """

    number_of_paths = x_0.size(0)
    D = x_0.size(1)
    S = config.number_of_states
    num_steps = config.number_of_steps
    min_t = 1./num_steps
    device = x_0.device

    with torch.no_grad():
        x = x_0

        ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
        save_ts = ts[np.linspace(0, len(ts)-2, config.num_intermediates, dtype=int)]

        if forward:
            ts = ts[::-1]
            save_ts = save_ts[::-1]

        x_hist = []
        x0_hist = []

        counter = 0
        for idx, t in tqdm(enumerate(ts[0:-1])):
            h = min_t
            times = t * torch.ones(number_of_paths,).to(device)
            reverse_rates = rate_model(x,times) # (N, D, S)
            x_0max = torch.max(reverse_rates, dim=2)[1]

            if t in save_ts:
                x_hist.append(x.clone().detach().cpu().numpy())
                x0_hist.append(x_0max.clone().detach().cpu().numpy())

            #TAU LEAPING
            diffs = torch.arange(S, device=device).view(1,1,S) - x.view(number_of_paths,D,1)
            poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
            jump_nums = poisson_dist.sample().to(device)
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = x + overall_jump
            x_new = torch.clamp(xp, min=0, max=S-1)

            x = x_new

        x_hist = np.array(x_hist).astype(int)
        x0_hist = np.array(x0_hist).astype(int)

        p_0gt = rate_model(x, min_t * torch.ones((number_of_paths,), device=device)) # (N, D, S)
        x_0max = torch.max(p_0gt, dim=2)[1]
        return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist


if __name__=="__main__":
    from torch.utils.data import TensorDataset,DataLoader
    from conditional_rate_matching.models.generative_models.crm import constant_rate
    from conditional_rate_matching.models.generative_models.crm import sample_categorical_from_dirichlet

    config = ConditionalRateMatchingConfig()
    config.number_of_states = 2
    config.gamma = 10.

    # Parameters
    dataset_0 = sample_categorical_from_dirichlet(probs=None,
                                                  alpha=config.dirichlet_alpha_0,
                                                  sample_size=config.sample_size,
                                                  dimension=config.number_of_spins,
                                                  number_of_states=config.number_of_states)
    tensordataset_0 = TensorDataset(dataset_0)
    dataloader_0 = DataLoader(tensordataset_0, batch_size=config.batch_size)

    dataset_1 = sample_categorical_from_dirichlet(probs=None,
                                                  alpha=config.dirichlet_alpha_1,
                                                  sample_size=183,
                                                  dimension=config.number_of_spins,
                                                  number_of_states=config.number_of_states)
    tensordataset_1 = TensorDataset(dataset_1)
    dataloader_1 = DataLoader(tensordataset_1, batch_size=config.batch_size)

    device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")

    databatch0 = next(dataloader_0.__iter__())
    x_0 = databatch0[0]
    x_0 = x_0.to(device)

    rate_model = lambda x,t: constant_rate(config,x,t)

    x_f,x_hist, x0_hist = TauLeaping(config,rate_model,x_0,forward=True)
    print(x_f.shape)
    print(dataset_1.shape)
    print(x_f)