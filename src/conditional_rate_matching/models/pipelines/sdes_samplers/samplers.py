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

def set_diagonal_rate(rates,x):
    """
    Ensures that we have the right diagonal rate
    """
    batch_size = rates.shape[0]
    dimensions = rates.shape[1]

    #set diagonal to sum of other values
    batch_index = torch.arange(batch_size).repeat_interleave((dimensions))
    dimension_index = torch.arange(dimensions).repeat((batch_size))
    rates[batch_index,dimension_index,x.long().view(-1)] = 0.

    #rate_diagonal = -rates.sum(axis=-1)
    #rates[batch_index,dimension_index,x.long().view(-1)] = rate_diagonal[batch_index,dimension_index]
    x_0max = torch.max(rates, dim=2)[1]
    #rates = rates * h
    #rates[batch_index,dimension_index,x.long().view(-1)] = 1. - rates[batch_index,dimension_index,x.long().view(-1)]

    #removes negatives
    #rates[torch.where(rates < 0.)]  = 0.

    return  x_0max,rates


def TauLeaping(config:Union[DSBConfig,CTDDConfig,CRMConfig],
               rate_model:Union[ClassificationForwardRate,SchrodingerBridgeRate],
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
    S = config.data0.vocab_size
    num_steps = config.pipeline.number_of_steps
    time_epsilon = config.pipeline.time_epsilon
    set_diagonal = config.pipeline.set_diagonal
    min_t = 1./num_steps
    device = x_0.device

    #==========================================
    # CONDITIONAL SAMPLING
    #==========================================
    conditional_tau_leaping = False
    conditional_model = False
    bridge_conditional = False
    if hasattr(config.data1,"conditional_model"):
        conditional_model = config.data1.conditional_model
        conditional_dimension = config.data1.conditional_dimension
        generation_dimension = config.data1.dimensions - conditional_dimension
        bridge_conditional = config.data1.bridge_conditional

    if conditional_model and not bridge_conditional:
        conditional_tau_leaping = True

    if conditional_tau_leaping:
        conditioner = x_0[:,0:conditional_dimension]

    with torch.no_grad():
        x = x_0

        ts = np.concatenate((np.linspace(1.0 - time_epsilon, min_t, num_steps), np.array([0])))

        if return_path:
            save_ts = np.concatenate((np.linspace(1.0 - time_epsilon, min_t, num_steps), np.array([0])))
        else:
            save_ts = ts[np.linspace(0, len(ts)-2, config.pipeline.num_intermediates, dtype=int)]

        if forward:
            ts = ts[::-1]
            save_ts = save_ts[::-1]

        x_hist = []
        x0_hist = []

        counter = 0
        for idx, t in tqdm(enumerate(ts[0:-1])):

            h = min_t
            times = t * torch.ones(number_of_paths,).to(device)
            rates = rate_model(x,times) # (N, D, S)
            x_0max = torch.max(rates, dim=2)[1]

            if t in save_ts:
                x_hist.append(x.clone().detach().unsqueeze(1))
                x0_hist.append(x_0max.clone().detach().unsqueeze(1))

            #TAU LEAPING
            diffs = torch.arange(S, device=device).view(1,1,S) - x.view(number_of_paths,D,1)
            poisson_dist = torch.distributions.poisson.Poisson(rates*h)
            jump_nums = poisson_dist.sample().to(device)
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=2)
            xp = x + overall_jump
            x_new = torch.clamp(xp, min=0, max=S-1)
            x = x_new

            # last step ------------------------------------------------
            if conditional_tau_leaping:
                x[:,0:conditional_dimension] = conditioner

        p_0gt = rate_model(x, min_t * torch.ones((number_of_paths,), device=device)) # (N, D, S)
        x_0max = torch.max(p_0gt, dim=2)[1]
        if conditional_tau_leaping:
            x_0max[:,0:conditional_dimension] = conditioner

        # save last step
        x_hist.append(x.clone().detach().unsqueeze(1))
        x0_hist.append(x_0max.clone().detach().unsqueeze(1))
        if len(x_hist) > 0:
            x_hist = torch.cat(x_hist,dim=1).float()
            x0_hist = torch.cat(x0_hist,dim=1).float()

        return x_0max.detach().float(), x_hist, x0_hist, torch.Tensor(save_ts.copy()).to(device)



def TauLeapingRates(config:Union[DSBConfig,CTDDConfig,CRMConfig],
                    rate_model:Union[ClassificationForwardRate,SchrodingerBridgeRate],
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
    S = config.data0.vocab_size
    num_steps = config.pipeline.number_of_steps
    min_t = 1./num_steps
    device = x_0.device

    with torch.no_grad():
        x = x_0
        ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))

        if return_path:
            save_ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
        else:
            save_ts = ts[np.linspace(0, len(ts)-2, config.pipeline.num_intermediates, dtype=int)]

        if forward:
            ts = ts[::-1]
            save_ts = save_ts[::-1]

        x_hist = []
        x0_hist = []
        rates_histogram = []

        counter = 0
        for idx, t in tqdm(enumerate(ts[0:-1])):

            h = min_t
            times = t * torch.ones(number_of_paths,).to(device)
            rates = rate_model(x,times) # (N, D, S)
            x_0max = torch.max(rates, dim=2)[1]

            if t in save_ts:
                x_hist.append(x.clone().detach().unsqueeze(1))
                x0_hist.append(x_0max.clone().detach().unsqueeze(1))
                rates_histogram.append(rates.clone().detach().unsqueeze(1))

            #TAU LEAPING
            diffs = torch.arange(S, device=device).view(1,1,S) - x.view(number_of_paths,D,1)
            poisson_dist = torch.distributions.poisson.Poisson(rates * h)
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
        rates_histogram.append(rates.clone().detach().unsqueeze(1))
        if len(x_hist) > 0:
            x_hist = torch.cat(x_hist,dim=1).float()
            x0_hist = torch.cat(x0_hist,dim=1).float()
            rates_histogram = torch.cat(rates_histogram, dim=1).float()

        return x_0max.detach().float(), x_hist, x0_hist, rates_histogram, torch.Tensor(save_ts.copy()).to(device)

"""
class ConditionalTauLeaping():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        eps_ratio = scfg.eps_ratio
        reject_multiple_jumps = scfg.reject_multiple_jumps
        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            ts = np.concatenate((np.linspace(1.0, min_t, num_steps), np.array([0])))
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            counter = 0
            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                qt0 = model.transition(t * torch.ones((N,), device=device)) # (N, S, S)
                rate = model.rate(t * torch.ones((N,), device=device)) # (N, S, S)

                model_input = torch.concat((conditioner, x), dim=1)
                p0t = F.softmax(model(model_input, t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                p0t = p0t[:, condition_dim:, :]


                x_0max = torch.max(p0t, dim=2)[1]
                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())



                qt0_denom = qt0[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N,sample_D,S) + eps_ratio

                # First S is x0 second S is x tilde

                qt0_numer = qt0 # (N, S, S)

                forward_rates = rate[
                    torch.arange(N, device=device).repeat_interleave(sample_D*S),
                    torch.arange(S, device=device).repeat(N*sample_D),
                    x.long().flatten().repeat_interleave(S)
                ].view(N, sample_D, S)

                inner_sum = (p0t / qt0_denom) @ qt0_numer # (N, D, S)

                reverse_rates = forward_rates * inner_sum # (N, D, S)

                reverse_rates[
                    torch.arange(N, device=device).repeat_interleave(sample_D),
                    torch.arange(sample_D, device=device).repeat(N),
                    x.long().flatten()
                ] = 0.0

                diffs = torch.arange(S, device=device).view(1,1,S) - x.view(N,sample_D,1)
                poisson_dist = torch.distributions.poisson.Poisson(reverse_rates * h)
                jump_nums = poisson_dist.sample()

                if reject_multiple_jumps:
                    jump_num_sum = torch.sum(jump_nums, dim=2)
                    jump_num_sum_mask = jump_num_sum <= 1
                    masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                    adj_diffs = masked_jump_nums * diffs
                else:
                    adj_diffs = jump_nums * diffs


                adj_diffs = jump_nums * diffs
                overall_jump = torch.sum(adj_diffs, dim=2)
                xp = x + overall_jump
                x_new = torch.clamp(xp, min=0, max=S-1)

                x = x_new

            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist
"""
"""
class ConditionalPCTauLeaping():
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, model, N, num_intermediates, conditioner):
        assert conditioner.shape[0] == N

        t = 1.0
        condition_dim = self.cfg.sampler.condition_dim
        total_D = self.cfg.data.shape[0]
        sample_D = total_D - condition_dim
        S = self.cfg.data.S
        scfg = self.cfg.sampler
        num_steps = scfg.num_steps
        min_t = scfg.min_t
        reject_multiple_jumps = scfg.reject_multiple_jumps
        eps_ratio = scfg.eps_ratio

        num_corrector_steps = scfg.num_corrector_steps
        corrector_step_size_multiplier = scfg.corrector_step_size_multiplier
        corrector_entry_time = scfg.corrector_entry_time

        initial_dist = scfg.initial_dist
        if initial_dist == 'gaussian':
            initial_dist_std  = model.Q_sigma
        else:
            initial_dist_std = None
        device = model.device

        with torch.no_grad():
            x = get_initial_samples(N, sample_D, device, S, initial_dist,
                initial_dist_std)


            h = 1.0 / num_steps # approximately
            ts = np.linspace(1.0, min_t+h, num_steps)
            save_ts = ts[np.linspace(0, len(ts)-2, num_intermediates, dtype=int)]

            x_hist = []
            x0_hist = []

            for idx, t in tqdm(enumerate(ts[0:-1])):
                h = ts[idx] - ts[idx+1]

                def get_rates(in_x, in_t):
                    qt0 = model.transition(in_t * torch.ones((N,), device=device)) # (N, S, S)
                    rate = model.rate(in_t * torch.ones((N,), device=device)) # (N, S, S)

                    model_input = torch.concat((conditioner, in_x), dim=1)
                    p0t = F.softmax(model(model_input, in_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
                    p0t = p0t[:, condition_dim:, :]


                    x_0max = torch.max(p0t, dim=2)[1]


                    qt0_denom = qt0[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        x.long().flatten().repeat_interleave(S)
                    ].view(N,sample_D,S) + eps_ratio

                    # First S is x0 second S is x tilde

                    qt0_numer = qt0 # (N, S, S)

                    forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        torch.arange(S, device=device).repeat(N*sample_D),
                        in_x.long().flatten().repeat_interleave(S)
                    ].view(N, sample_D, S)

                    reverse_rates = forward_rates * ((p0t/qt0_denom) @ qt0_numer) # (N, D, S)

                    reverse_rates[
                        torch.arange(N, device=device).repeat_interleave(sample_D),
                        torch.arange(sample_D, device=device).repeat(N),
                        in_x.long().flatten()
                    ] = 0.0

                    transpose_forward_rates = rate[
                        torch.arange(N, device=device).repeat_interleave(sample_D*S),
                        in_x.long().flatten().repeat_interleave(S),
                        torch.arange(S, device=device).repeat(N*sample_D)
                    ].view(N, sample_D, S)

                    return transpose_forward_rates, reverse_rates, x_0max

                def take_poisson_step(in_x, in_reverse_rates, in_h):
                    diffs = torch.arange(S, device=device).view(1,1,S) - in_x.view(N,sample_D,1)
                    poisson_dist = torch.distributions.poisson.Poisson(in_reverse_rates * in_h)
                    jump_nums = poisson_dist.sample()

                    if reject_multiple_jumps:
                        jump_num_sum = torch.sum(jump_nums, dim=2)
                        jump_num_sum_mask = jump_num_sum <= 1
                        masked_jump_nums = jump_nums * jump_num_sum_mask.view(N, sample_D, 1)
                        adj_diffs = masked_jump_nums * diffs
                    else:
                        adj_diffs = jump_nums * diffs

                    overall_jump = torch.sum(adj_diffs, dim=2)
                    xp = in_x + overall_jump
                    x_new = torch.clamp(xp, min=0, max=S-1)
                    return x_new

                transpose_forward_rates, reverse_rates, x_0max = get_rates(x, t)

                if t in save_ts:
                    x_hist.append(x.clone().detach().cpu().numpy())
                    x0_hist.append(x_0max.clone().detach().cpu().numpy())

                x = take_poisson_step(x, reverse_rates, h)
                if t <= corrector_entry_time:
                    for _ in range(num_corrector_steps):
                        transpose_forward_rates, reverse_rates, _ = get_rates(x, t-h)
                        corrector_rate = transpose_forward_rates + reverse_rates
                        corrector_rate[
                            torch.arange(N, device=device).repeat_interleave(sample_D),
                            torch.arange(sample_D, device=device).repeat(N),
                            x.long().flatten()
                        ] = 0.0
                        x = take_poisson_step(x, corrector_rate,
                            corrector_step_size_multiplier * h)



            x_hist = np.array(x_hist).astype(int)
            x0_hist = np.array(x0_hist).astype(int)

            model_input = torch.concat((conditioner, x), dim=1)
            p_0gt = F.softmax(model(model_input, min_t * torch.ones((N,), device=device)), dim=2) # (N, D, S)
            p_0gt = p_0gt[:, condition_dim:, :]
            x_0max = torch.max(p_0gt, dim=2)[1]
            output = torch.concat((conditioner, x_0max), dim=1)
            return output.detach().cpu().numpy().astype(int), x_hist, x0_hist

"""