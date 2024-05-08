import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from torch import functional as F

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.configs_classes.config_dsb import DSBConfig
from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig

from conditional_rate_matching.models.temporal_networks.rates.dsb_rate import SchrodingerBridgeRate
from conditional_rate_matching.models.temporal_networks.rates.crm_rates import ClassificationForwardRate



def PCTauLeaping(config:Union[DSBConfig,CTDDConfig,CRMConfig],
                 rate_model:Union[ClassificationForwardRate,SchrodingerBridgeRate],
                 x_0:torch.Tensor,
                 forward=True,
                 return_path=False):
    t = 1.0
    N = x_0.size(0)
    D = x_0.size(1)
    S = config.data0.vocab_size
    num_steps = config.pipeline.number_of_steps
    time_epsilon = config.pipeline.time_epsilon
    min_t = 1./num_steps
    device = x_0.device
    num_intermediates = config.pipeline.num_intermediates

    num_corrector_steps = config.pipeline.num_corrector_steps
    corrector_step_size_multiplier = config.pipeline.corrector_step_size_multiplier
    corrector_entry_time = config.pipeline.corrector_entry_time


    with torch.no_grad():
        h = 1.0 / num_steps  # approximately
        ts = np.linspace(1.0, min_t + h, num_steps)
        save_ts = ts[np.linspace(0, len(ts) - 2, num_intermediates, dtype=int)]

        x_hist = []
        x0_hist = []

        for idx, t in tqdm(enumerate(ts[0:-1])):

            h = ts[idx] - ts[idx + 1]

            def get_rates(in_x, in_t):
                qt0 = model.transition(in_t * torch.ones((N,), device=device))  # (N, S, S)
                rate = model.rate(in_t * torch.ones((N,), device=device))  # (N, S, S)

                p0t = F.softmax(model(in_x, in_t * torch.ones((N,), device=device)), dim=2)  # (N, D, S)

                x_0max = torch.max(p0t, dim=2)[1]

                qt0_denom = qt0[
                                torch.arange(N, device=device).repeat_interleave(D * S),
                                torch.arange(S, device=device).repeat(N * D),
                                in_x.long().flatten().repeat_interleave(S)
                            ].view(N, D, S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0  # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(D * S),
                    torch.arange(S, device=device).repeat(N * D),
                    in_x.long().flatten().repeat_interleave(S)
                ].view(N, D, S)

                reverse_rates = forward_rates * ((p0t / qt0_denom) @ qt0_numer)  # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(D),
                    torch.arange(D, device=device).repeat(N),
                    in_x.long().flatten()
                ] = 0.0

                transpose_forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(D * S),
                    in_x.long().flatten().repeat_interleave(S),
                    torch.arange(S, device=device).repeat(N * D)
                ].view(N, D, S)

                return transpose_forward_rates, reverse_rates, x_0max

            def take_poisson_step(in_x, in_reverse_rates, in_h):
                diffs = torch.arange(S, device=device).view(1, 1, S) - in_x.view(N, D, 1)
                poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                jump_nums = poisson_dist.sample()
                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                unclip_x_new = in_x + overall_jump
                x_new = torch.clamp(unclip_x_new, min=0, max=S - 1)

                return x_new

            transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

            if t in save_ts:
                x_hist.append(x.detach().cpu().numpy())
                x0_hist.append(x_0max.detach().cpu().numpy())

            x = take_poisson_step(x, reverse_rates, h)

            if t <= corrector_entry_time:
                for _ in range(num_corrector_steps):
                    transpose_forward_rates, reverse_rates, _ = get_rates(x, t - h)
                    corrector_rate = transpose_forward_rates + reverse_rates
                    corrector_rate[
                        torch.arange(N, device=device).repeat_interleave(D),
                        torch.arange(D, device=device).repeat(N),
                        x.long().flatten()
                    ] = 0.0
                    x = take_poisson_step(x, corrector_rate,
                                          corrector_step_size_multiplier * h)

        x_hist = np.array(x_hist).astype(int)
        x0_hist = np.array(x0_hist).astype(int)

        p_0gt = F.softmax(model(x, min_t * torch.ones((N,), device=device)), dim=2)  # (N, D, S)
        x_0max = torch.max(p_0gt, dim=2)[1]

        return x_0max.detach().cpu().numpy().astype(int), x_hist, x0_hist