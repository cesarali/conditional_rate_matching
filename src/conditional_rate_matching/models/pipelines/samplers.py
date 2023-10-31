import torch
import numpy as np
from tqdm import tqdm
from torch import functional as F
from typing import Union,Tuple,List

from conditional_rate_matching.configs.config_crm import Config as ConditionalRateMatchingConfig
from conditional_rate_matching.models.temporal_networks.backward_rates.crm_backward_rates import ConditionalBackwardRate



def TauLeaping(config:ConditionalRateMatchingConfig,
               rate_model:Union[ConditionalBackwardRate],
               x_0:torch.Tensor,
               forward=True,
               return_path=False):
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

        if return_path:
            save_ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
        else:
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
                x_hist.append(x.clone().detach().unsqueeze(1))
                x0_hist.append(x_0max.clone().detach().unsqueeze(1))

            #TAU LEAPING
            diffs = torch.arange(S, device=device).view(1,1,S) - x.view(number_of_paths,D,1)
            poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
            jump_nums = poisson_dist.sample().to(device)
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = x + overall_jump
            x_new = torch.clamp(xp, min=0, max=S-1)

            x = x_new

        # last step
        p_0gt = rate_model(x, min_t * torch.ones((number_of_paths,), device=device)) # (N, D, S)
        x_0max = torch.max(p_0gt, dim=2)[1]

        # save last step
        x_hist.append(x.clone().detach().unsqueeze(1))
        x0_hist.append(x_0max.clone().detach().unsqueeze(1))
        if len(x_hist) > 0:
            x_hist = torch.cat(x_hist,dim=1)
            x0_hist = torch.cat(x0_hist,dim=1)

        return x_0max.detach(), x_hist, x0_hist, torch.Tensor(ts.copy())