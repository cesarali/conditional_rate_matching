import torch
from tqdm import tqdm
from conditional_rate_matching.models.generative_models.crm import CRM

def telegram_bridge_probability_path(crm:CRM,ts,x_1,x_0):
    x_to_go = crm.forward_rate.where_to_go_x(x_1)
    # TELEGRAM PROBABILITY
    probability_path = []
    for time in tqdm(ts):
        telegram_probability = crm.forward_rate.telegram_bridge_probability(x_to_go, x_1, x_0, time)
        telegram_probability = telegram_probability.mean(dim=0).unsqueeze(dim=0)
        probability_path.append(telegram_probability)
    probability_path = torch.cat(probability_path,dim=0)
    return probability_path

def telegram_bridge_sample_paths(crm:CRM,X_0,X_1,time_steps,histogram=True):
    sample = []
    for time_value in time_steps:
        time_ = torch.full((X_0.size(0),),time_value)
        sampled_x = crm.forward_rate.sample_x(X_1, X_0, time_)
        if histogram:
            histogram_t = sampled_x.sum(axis=0)
            sample.append(histogram_t.unsqueeze(0))
        else:
            sample.append(sampled_x)
    sample = torch.cat(sample,dim=0)
    return sample,time_steps

# CONDITIONAL TRANSITION RATE
def conditional_transition_rate_path(crm:CRM,ts,x_1):
    x_to_go = crm.forward_rate.where_to_go_x(x_1)
    rate_path = []
    for time in tqdm(ts):
        bridge_rate = crm.forward_rate.conditional_transition_rate(x_to_go, x_1, time)
        bridge_rate = bridge_rate.mean(dim=0).unsqueeze(dim=0)
        rate_path.append(bridge_rate)
    rate_path = torch.cat(rate_path,dim=0)
    return rate_path

def conditional_probability_path(crm:CRM,ts,x_0):
    x_to_go = crm.forward_rate.where_to_go_x(x_0)
    rate_path = []
    for time in tqdm(ts):
        bridge_rate = crm.forward_rate.conditional_probability(x_to_go, x_0, time.item() , t0=0.)
        bridge_rate = bridge_rate.mean(dim=0).unsqueeze(dim=0)
        rate_path.append(bridge_rate)
    rate_path = torch.cat(rate_path,dim=0)
    return rate_path

"""
def classification_path(model,x_1,ts):
    device = x_1.device
    # MODEL RATE
    rate_path = []
    for time in tqdm(ts):
        time_ = torch.full((x_1.size(0),), time.item()).to(device)
        bridge_rate = model.classify(x_1, time_)
        bridge_rate = bridge_rate.mean(dim=0).unsqueeze(dim=0)
        rate_path.append(bridge_rate)
    rate_path = torch.cat(rate_path, dim=0)
    return rate_path
"""

def classification_path(model, x_1, ts, batch_size=32):
    with torch.no_grad():
        device = x_1.device
        # MODEL RATE
        rate_path = []

        # If batch_size is None or x_1 is within the batch limit, process normally
        if batch_size is None or x_1.size(0) <= batch_size:
            for time in tqdm(ts):
                time_ = torch.full((x_1.size(0),), time.item()).to(device)
                bridge_rate = model.classify(x_1, time_)
                bridge_rate = bridge_rate.mean(dim=0).unsqueeze(dim=0)
                rate_path.append(bridge_rate)
        else:
            # Process in batches if x_1 is larger than batch_size
            for time in tqdm(ts):
                # Initialize temporary list to store batch results for current time
                temp_rate_path = []
                for i in range(0, x_1.size(0), batch_size):
                    x_1_batch = x_1[i:i + batch_size]
                    time_ = torch.full((x_1_batch.size(0),), time.item()).to(device)
                    bridge_rate = model.classify(x_1_batch, time_)
                    # Note that we're taking the mean over dim=0 for each batch separately
                    bridge_rate = bridge_rate.mean(dim=0)
                    temp_rate_path.append(bridge_rate.unsqueeze(0))

                # Concatenate the batch results for current time and add to rate_path
                batched_rate = torch.cat(temp_rate_path,dim=0).mean(dim=0)
                rate_path.append(batched_rate.unsqueeze(0))

        # Concatenate along time dimension to get the full rate path
        rate_path = torch.cat(rate_path, dim=0)
        return rate_path
