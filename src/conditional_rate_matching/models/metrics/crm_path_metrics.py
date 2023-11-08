import torch
from tqdm import tqdm

from conditional_rate_matching.models.generative_models.crm import (
    telegram_bridge_probability,
    conditional_transition_rate,
    conditional_probability,
    sample_x,
    where_to_go_x
)

def telegram_bridge_probability_path(config,ts,x_1,x_0):
    x_to_go = where_to_go_x(config,x_1)
    # TELEGRAM PROBABILITY
    probability_path = []
    for time in tqdm(ts):
        telegram_probability = telegram_bridge_probability(config, x_to_go, x_1, x_0, time)
        telegram_probability = telegram_probability.mean(dim=0).unsqueeze(dim=0)
        probability_path.append(telegram_probability)
    probability_path = torch.cat(probability_path,dim=0)
    return probability_path

def telegram_bridge_sample_paths(config,X_0,X_1,time_steps,histogram=True):
    sample = []
    for time_value in time_steps:
        time_ = torch.full((X_0.size(0),),time_value)
        sampled_x = sample_x(config, X_1, X_0, time_)
        if histogram:
            histogram_t = sampled_x.sum(axis=0)
            sample.append(histogram_t.unsqueeze(0))
        else:
            sample.append(sampled_x)
    sample = torch.cat(sample,dim=0)
    return sample,time_steps

# CONDITIONAL TRANSITION RATE
def conditional_transition_rate_path(config,ts,x_1):
    x_to_go = where_to_go_x(config,x_1)
    rate_path = []
    for time in tqdm(ts):
        bridge_rate = conditional_transition_rate(config, x_to_go, x_1, time)
        bridge_rate = bridge_rate.mean(dim=0).unsqueeze(dim=0)
        rate_path.append(bridge_rate)
    rate_path = torch.cat(rate_path,dim=0)
    return rate_path

def conditional_probability_path(config,ts,x_0):
    x_to_go = where_to_go_x(config,x_0)
    rate_path = []
    for time in tqdm(ts):
        bridge_rate = conditional_probability(config, x_to_go, x_0, time.item() , t0=0.)
        bridge_rate = bridge_rate.mean(dim=0).unsqueeze(dim=0)
        rate_path.append(bridge_rate)
    rate_path = torch.cat(rate_path,dim=0)
    return rate_path

def classification_path(model,x_1,ts):
    device = x_1.device
    # MODEL RATE
    rate_path = []
    for time in tqdm(ts):
        print(f"time {time}")
        time_ = torch.full((x_1.size(0),), time.item()).to(device)
        bridge_rate = model.classify(x_1, time_)
        bridge_rate = bridge_rate.mean(dim=0).unsqueeze(dim=0)
        rate_path.append(bridge_rate)
    rate_path = torch.cat(rate_path, dim=0)
    return rate_path
