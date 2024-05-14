import os
import torch
import numpy as np
from dataclasses import dataclass, asdict
import torch.nn.functional as F
from conditional_rate_matching.models.generative_models.crm import CRM
from torch.distributions import Categorical

def rate_to_probabilities(rate_f,x_0,x_1,delta_t):
    """
    """
    # find who is equal
    batch_index_equal = torch.where(x_0 == x_1)[0]
    dimension_equal = torch.where(x_0 == x_1)[1]
    who_equal = x_0[torch.where(x_0 == x_1)].long()

    # set diagonal to one
    rate_f[batch_index_equal,dimension_equal,who_equal] = 0.
    rates_diagonal = rate_f.sum(axis=-1)
    rate_f[batch_index_equal,dimension_equal,who_equal] = -rates_diagonal[batch_index_equal,dimension_equal]

    #calculate probabilities
    rate_f = delta_t*rate_f
    rate_f[batch_index_equal,dimension_equal,who_equal] = 1. + rate_f[batch_index_equal,dimension_equal,who_equal]
    return rate_f 

def evaluate_rate_in_batches(crm,crm_b,x_path_b_0,x_path_b_1,time_b_0,time_f_1,batch_size=10,shapes=(32,10,2)):
    """
    """
    sample_size,number_of_time_steps,dimensions = shapes
    total_batches = (sample_size * (number_of_time_steps - 1) + batch_size - 1) // batch_size

    # Initialize empty lists to store batch results if necessary
    rates_b = []
    rates_f = []
    for batch_index in range(total_batches):
        start_idx = batch_index * batch_size
        end_idx = min((batch_index + 1) * batch_size, sample_size * (number_of_time_steps - 1))

        # Extracting batches for input to the neural network
        x_path_b_0_batch = x_path_b_0[start_idx:end_idx]
        x_path_b_1_batch = x_path_b_1[start_idx:end_idx]
        time_b_0_batch = time_b_0[start_idx:end_idx]
        time_f_1_batch = time_f_1[start_idx:end_idx]

        # Evaluating forward and backward rates for each batch
        rate_b_batch = crm_b.forward_rate(x_path_b_0_batch, time_b_0_batch)
        rate_f_batch = crm.forward_rate(x_path_b_1_batch, time_f_1_batch)

        # Storing the batch results if necessary for further processing
        rates_b.append(rate_b_batch)
        rates_f.append(rate_f_batch)
    
    # Concatenate all batch results into a single tensor
    rates_b = torch.cat(rates_b, dim=0)
    rates_f = torch.cat(rates_f, dim=0)

    return rates_b,rates_f

@torch.no_grad
def calculate_batch_log_likelihood(crm:CRM,crm_b:CRM,databatch1,delta_t=None,ignore_=1,in_batches=False,batch_size=100):
    """
    Parameters
    ----------
    crm: trained crm model in forward time
    crm_b: trained crm model in forward time
    databatch1: databacth to calculate likelihood with
    delta_t: None
    ignore_: steps to ignore of the process

    y0 corresponds to the variables indexing of the backward process

    q(y3|y2)q(y2|y1)q(y1|y0) q(y0) <- q is backward process 

    numerator    q(x0|x1)q(x1|x2)q(x2|x3)  q(x3)
    denominator  p(x1|x0)p(x2|x1)p(x3|x2)

    q(3) is sample from target of forward process

    y0 corresponds to x3 for the forward process y3 to x0

    returns
    -------
    x_0, log_likelihood_10: 
        (torch.Tensor) from backward process x_0
        (torch.Tensor) log_likelihood per batch (dimensions sum out)
    """
    x_1 = databatch1[0]
    batch_size = x_1.shape[0]

    # we simulate the path backwards
    x_f, x_path_b, t_path_b = crm_b.pipeline(return_path=True,x_0=x_1)
    sample_size, number_of_time_steps, dimensions = x_path_b.shape[0],x_path_b.shape[1],x_path_b.shape[2]
    t_path_b = t_path_b[None,:].repeat(batch_size,1)

    # reshape states and vectors of paths (one step ahead)
    # we calculate the inverted time (the backward process was also trained 0. to 1.)
    x_0 = x_path_b[:,-(ignore_+1),:]
    x_path_b_0 = x_path_b[:,:-1,:]
    time_b_0 = t_path_b[:,:-1] 

    x_path_b_1 = x_path_b[:,1:,:]
    time_f_1 = 1. - t_path_b[:,1:]

    x_path_b_0 = x_path_b_0.reshape(sample_size*(number_of_time_steps-1),dimensions)
    x_path_b_1 = x_path_b_1.reshape(sample_size*(number_of_time_steps-1),dimensions)
    time_b_0 = time_b_0.reshape(sample_size*(number_of_time_steps-1)) # reshape
    time_f_1 = time_f_1.reshape(sample_size*(number_of_time_steps-1)) # reshape

    if delta_t is None:
        delta_t = (t_path_b[:,1:] -  t_path_b[:,:-1])[0,3]

    # we evaluate forward and backward rate in backward process
    if in_batches:
        shapes = (sample_size,number_of_time_steps,dimensions)
        rate_b,rate_f = evaluate_rate_in_batches(crm,crm_b,x_path_b_0,x_path_b_1,time_b_0,time_f_1,
                                                 batch_size=batch_size,shapes=shapes)
    else:
        rate_b = crm_b.forward_rate(x_path_b_0,time_b_0)
        rate_f = crm.forward_rate(x_path_b_1,time_f_1)

    # we convert to probabilities based on the rule for the rates diagonal
    rate_f = rate_to_probabilities(rate_f,x_path_b_0,x_path_b_1,delta_t)
    rate_b = rate_to_probabilities(rate_b,x_path_b_0,x_path_b_1,delta_t)

    # we take the values of the rates evaluating at the correspondig next step
    rate_b = torch.gather(rate_b,2,x_path_b_1[:,:,None].long()).squeeze()
    rate_f = torch.gather(rate_f,2,x_path_b_0[:,:,None].long()).squeeze()

    # we write everything in the time dimensions
    rate_b = rate_b.reshape((batch_size,number_of_time_steps-1,dimensions))
    rate_f = rate_f.reshape((batch_size,number_of_time_steps-1,dimensions))
    
    # calculate log likelihood
    log_1_0 = torch.log(rate_b) - torch.log(rate_f)
    log_1_0 = log_1_0[:,ignore_:-ignore_,:]
    log_1_0 = log_1_0.sum(axis=1) # sum over time
    log_1_0 = log_1_0.sum(axis=-1) # sum over dimensions

    return x_0,log_1_0

def get_log_likelihood(crm:CRM,crm_b:CRM,delta_t=None,ignore_=1,in_batches=False):
    """
    """
    dimensions = crm.config.data0.dimensions
    vocab_size = crm.config.data0.vocab_size
    probabilities_0 = torch.ones((dimensions,vocab_size))/vocab_size
    x0_distribution = Categorical(probabilities_0)

    LOG = 0.
    sample_size = 0
    for databatch1 in crm.dataloader_1.test():#<----------------------

        x_0, log_10 = calculate_batch_log_likelihood(crm,crm_b,databatch1,delta_t=delta_t,ignore_=ignore_,in_batches=in_batches)
        log_0 = x0_distribution.log_prob(x_0).sum(axis=-1)
        
        log_1 = log_10 - log_0

        # average overall data set
        batch_size = x_0.size(0)
        sample_size += batch_size
        LOG = log_1.sum()
    LOG = LOG/sample_size
    return -LOG

def get_log_likelihood_states_dataloader(crm:CRM):
    """
    Log probabilities of known categorical probabilities
    """
    probabilities_1 = torch.tensor(crm.config.data1.bernoulli_probability).squeeze()
    x1_distribution = Categorical(probabilities_1)
    sample_size = 0
    LOG = 0.
    for databatch1 in crm.dataloader_1.test():
        x1 = databatch1[0]
        log_1 = x1_distribution.log_prob(x1).sum(axis=-1)

        # average overall data set
        batch_size = x1.size(0)
        sample_size += batch_size
        LOG = log_1.sum()
    LOG = LOG/sample_size
    return LOG


