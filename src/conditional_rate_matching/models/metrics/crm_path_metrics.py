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
    """
    for a given end point, we average over the rate values
    """
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

from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping,TauLeapingRates

def conditional_bridge_marginal_probabilities_and_rates_path(crm,
                                                             steps_of_noise_to_see=500,
                                                             number_of_steps=1000,
                                                             max_sample_size=300):
    """
    returns
    -------
    rate_average_per_time,rate_average_per_time_1,histogram_per_dimension
    """
    config = crm.config

    config.pipeline.num_intermediates = steps_of_noise_to_see
    config.pipeline.number_of_steps = number_of_steps

    rate_average_per_time = torch.zeros(steps_of_noise_to_see, config.data0.dimensions)
    rate_average_per_time_1 = torch.zeros(steps_of_noise_to_see, config.data0.dimensions)

    histogram_per_dimension = torch.zeros(steps_of_noise_to_see, config.data0.dimensions)

    sample_size = 0
    for databatch_0, databatch_1 in zip(crm.dataloader_0.test(), crm.dataloader_1.test()):
        x_0 = databatch_0[0]
        x_1 = databatch_1[0]
        sample_size += x_0.size(0)

        rate_model = lambda x, t: crm.forward_rate.conditional_transition_rate(x, x_1, t)
        x_f, x_hist, x0_hist, rates_histogram, ts = TauLeapingRates(config, rate_model, x_0, forward=True)

        rate_average_per_time += rates_histogram.sum(axis=0)[:, :, 0]
        rate_average_per_time_1 += rates_histogram.sum(axis=0)[:, :, 1]

        histogram_per_dimension += x_hist.sum(axis=0)

        if sample_size > max_sample_size:
            break

    rate_average_per_time = rate_average_per_time / sample_size
    rate_average_per_time_1 = rate_average_per_time_1 / sample_size

    histogram_per_dimension = histogram_per_dimension / sample_size

    return rate_average_per_time,rate_average_per_time_1,histogram_per_dimension,ts

def conditional_bridge_images(crm,
                              x_0,
                              x_1,
                              steps_of_noise_to_see=500,
                              number_of_steps=1000):
    config = crm.config

    config.pipeline.num_intermediates = steps_of_noise_to_see
    config.pipeline.number_of_steps = number_of_steps

    rate_model = lambda x, t: crm.forward_rate.conditional_transition_rate(x, x_1, t)
    x_f, x_hist, x0_hist, ts = TauLeaping(config, rate_model, x_0, forward=True)
    return x_hist,ts