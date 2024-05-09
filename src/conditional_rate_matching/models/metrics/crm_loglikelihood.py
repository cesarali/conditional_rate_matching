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

def calculate_batch_log_likelihood(crm:CRM,crm_b:CRM,databatch1,delta_t=None,ignore_=1, device='cpu'):
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
    
    x_1 = databatch1[0].to(device)
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

@torch.no_grad()
def get_log_likelihood(crm,crm_b,delta_t=None,ignore_=1, device='cpu'):
    """
    """
    dimensions = crm.config.data0.dimensions
    vocab_size = crm.config.data0.vocab_size
    probabilities_0 = torch.ones((dimensions,vocab_size))/vocab_size

    x0_distribution = Categorical(probabilities_0)

    LOG = 0.
    sample_size = 0
    for databatch1 in crm.dataloader_1.test():#<----------------------
        x_0, log_10 = calculate_batch_log_likelihood(crm,crm_b,databatch1,delta_t=delta_t,ignore_=ignore_, device=device)
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

if __name__=="__main__":
    print("Hey!")

    