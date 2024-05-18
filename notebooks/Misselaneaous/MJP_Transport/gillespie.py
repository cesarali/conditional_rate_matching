import os
import sys
import torch
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from typing import Tuple,List
from dataclasses import dataclass
from matplotlib import pyplot as plt
from torch.distributions import Exponential,Categorical,Dirichlet

@dataclass
class DirichletPriorOnRatesParam:
    concentration:float = 1.
    num_states:int = 5

def generate_transition_rate_matrix(parameters:DirichletPriorOnRatesParam):
    """
    Generate a transition rate matrix for a Markov jump process.
    
    Parameters:
        num_states (int): Number of states in the Markov process.
        concentration (float or list): Concentration parameter(s) for the Dirichlet distribution. 
                                      Higher values lead to more uniform distributions.

    Returns:
        torch.Tensor: A transition rate matrix of shape (num_states, num_states).
    """
    num_states = parameters.num_states
    concentration = parameters.concentration

    # Initialize a tensor to store the transition rate matrix
    rate_matrix = torch.zeros(num_states, num_states)

    # Generate rows from Dirichlet distributions
    for i in range(num_states):
        # Create concentration parameters for the Dirichlet distribution
        # Each row should have num_states - 1 positive entries (excluding the diagonal)
        if isinstance(concentration, float):
            alpha = torch.full((num_states - 1,), concentration)
        else:
            alpha = torch.tensor(concentration[i])
        
        # Exclude diagonal element by creating a temporary row and then inserting
        temp_row = Dirichlet(alpha).sample()
        rate_matrix[i, :i] = temp_row[:i]
        rate_matrix[i, i+1:] = temp_row[i:]

        # Set the diagonal element such that the sum of the row is zero
        rate_matrix[i, i] = -torch.sum(rate_matrix[i])

    return rate_matrix

@dataclass
class LotkaVolterraParameters:
    alpha:float = 0.0005
    beta:float = 0.0001
    delta:float = 0.0001
    gamma:float = 0.0005

def lotka_volterra_rate(XY0,parameters:LotkaVolterraParameters):
    """
    parameters
    ----------
    X: torch.Tensor(number_of_paths,2) 

    returns
    -------
    new_states: torch.Tensor(number_of_paths,2,4)
    rates: torch.Tensor(number_of_paths,4) 
    """
    number_of_paths = XY0.size(0)
    #where_empty_population = torch.where(XY0 == 0)
    #XY0[where_empty_population] = 1e-6
    
    # new states available in next step
    new_states_mask = torch.Tensor([[1.,0],
                                    [-1.,0],
                                    [0.,1.],
                                    [0.,-1.]])

    mask_per_path = new_states_mask[None,:].repeat((number_of_paths,1,1))
    mask_per_path = mask_per_path.permute((0,2,1))

    #rates for new states available in next step
    prey = XY0[:,0].clone()
    predator = XY0[:,1].clone()

    X_up = parameters.alpha*prey
    X_down = parameters.beta*prey*predator

    Y_up = parameters.delta*prey*predator
    Y_down = parameters.gamma*predator

    Y_up[torch.where(Y_up == 0.)] = 1e-3
    X_up[torch.where(X_up == 0.)] = 1e-3

    rates = torch.cat([X_up[:,None],X_down[:,None],Y_up[:,None],Y_down[:,None]],dim=-1)
    #rates[][where_empty_population]= 0.001

    if torch.any(rates < 0.):
        where_negative_rate = torch.where(rates < 0.)
        print(rates[where_negative_rate])
        print(predator[where_negative_rate[0]])
        print(prey[where_negative_rate[0]])

    return mask_per_path,rates

@dataclass
class RepressilatorParameters:
    # Parameter values
    kmu:float = 0.5
    kmo:float = 5e-4
    kp:float = 0.167
    gamma_m:float = 0.005776
    gamma_p:float = 0.001155
    kr:float = 1.0
    ku1:float = 224.0
    ku2:float = 9.0

def repressilator_mask():
    repressilator_update = np.array([
    # 0   1   2   3   4   5   6   7   8
    [ 1,  0,  0,  0,  0,  0,  0,  0,  0], # 0
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0], # 1
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0], # 2
    [ 0,  1,  0,  0,  0,  0,  0,  0,  0], # 3
    [ 0,  0,  0,  1,  0,  0,  0,  0,  0], # 4
    [ 0,  0,  0,  0,  0,  1,  0,  0,  0], # 5
    [-1,  0,  0,  0,  0,  0,  0,  0,  0], # 6
    [ 0,  0, -1,  0,  0,  0,  0,  0,  0], # 7
    [ 0,  0,  0,  0, -1,  0,  0,  0,  0], # 8
    [ 0, -1,  0,  0,  0,  0,  0,  0,  0], # 9
    [ 0,  0,  0, -1,  0,  0,  0,  0,  0], # 10
    [ 0,  0,  0,  0,  0, -1,  0,  0,  0], # 11
    [ 0,  0,  0,  0,  0,  0, -1,  0,  0], # 12
    [ 0,  0,  0,  0,  0,  0,  0, -1,  0], # 13
    [ 0,  0,  0,  0,  0,  0,  0,  0, -1], # 14
    [ 0,  0,  0,  0,  0, -1,  0,  0,  1], # 15
    [ 0, -1,  0,  0,  0,  0,  1,  0,  0], # 16
    [ 0,  0,  0, -1,  0,  0,  0,  1,  0], # 17
    [ 0,  0,  0,  0,  0,  1,  0,  0, -1], # 18
    [ 0,  1,  0,  0,  0,  0, -1,  0,  0], # 19
    [ 0,  0,  0,  1,  0,  0,  0, -1,  0], # 20
    ], dtype=float)
    return torch.Tensor(repressilator_update)

def repressilator_rate(population,params:RepressilatorParameters):
    number_of_paths = population.size(0)
    new_states_mask = repressilator_mask()
    mask_per_path = new_states_mask[None,:].repeat((number_of_paths,1,1))
    mask_per_path = mask_per_path.permute((0,2,1))

    # Extract each component from the population tensor
    m1, p1, m2, p2, m3, p3, n1, n2, n3 = population[:, 0], population[:, 1], population[:, 2], population[:, 3], population[:, 4], population[:, 5], population[:, 6], population[:, 7], population[:, 8]
    
    # Initialize propensities tensor
    propensities = torch.zeros(population.shape[0], 21)
    
    # Update propensities based on model logic
    propensities[:, 0] = torch.where(n3 == 0, params.kmu, params.kmo)
    propensities[:, 1] = torch.where(n1 == 0, params.kmu, params.kmo)
    propensities[:, 2] = torch.where(n2 == 0, params.kmu, params.kmo)
    propensities[:, 3] = params.kp * m1
    propensities[:, 4] = params.kp * m2
    propensities[:, 5] = params.kp * m3
    propensities[:, 6] = params.gamma_m * m1
    propensities[:, 7] = params.gamma_m * m2
    propensities[:, 8] = params.gamma_m * m3
    propensities[:, 9] = params.gamma_p * p1
    propensities[:, 10] = params.gamma_p * p2
    propensities[:, 11] = params.gamma_p * p3
    propensities[:, 12] = params.gamma_p * n1
    propensities[:, 13] = params.gamma_p * n2
    propensities[:, 14] = params.gamma_p * n3
    propensities[:, 15] = params.kr * p3 * (n3 < 2)
    propensities[:, 16] = params.kr * p1 * (n1 < 2)
    propensities[:, 17] = params.kr * p2 * (n2 < 2)
    propensities[:, 18] = params.ku1 * (n3 == 1) + 2 * params.ku2 * (n3 == 2)
    propensities[:, 19] = params.ku1 * (n1 == 1) + 2 * params.ku2 * (n1 == 2)
    propensities[:, 20] = params.ku1 * (n2 == 1) + 2 * params.ku2 * (n2 == 2)
    
    return mask_per_path,propensities

def choose_new_states(new_states_available,which_state_to_take):
    """
    selects from available according to which_state_to_take 

    parameters
    ----------

    new_states_available: (number_of_paths,dimension,number_of_new_possible)
    which_state_to_take :(number_of_paths) \in [0,number_of_new_possible]

    returns
    -------
    new_states: (number_of_paths,dimension)
    """
    # Assuming N, D, and number_of_states are defined
    N = new_states_available.size(0)  # example value
    D = new_states_available.size(1)   # example value
    # To use selected_index to index X, we need to unsqueeze it to make it broadcastable
    which_state_to_take = which_state_to_take.unsqueeze(-1).expand(N, D)  # Shape (N, D)
    # Now, gather the elements. We want to gather along the last dimension
    new_states = torch.gather(new_states_available, 2, which_state_to_take.unsqueeze(2)).squeeze(2)  # Use unsqueeze to match the gather requirement and squeeze to drop the extra dimension
    
    return new_states

def gillespie_mask(X0,rate_function,number_of_times=100)->Tuple[torch.Tensor,torch.Tensor]:
    """
    # http://be150.caltech.edu/2019/handouts/12_stochastic_simulation_all_code.html

    parameters
    ----------

    X0: initial conditions
    rate_function: function -> new_states_available,rates

    returns
    -------
    paths,times
    """
    batch_size = X0.shape[0]

    # Initialize process
    times = torch.full((batch_size, 1), 0.)
    paths = X0.unsqueeze(1)

    for time_index in range(number_of_times):
        X = paths[:,-1,:]

        #if torch.any(X[:,0] > 100):
        #    where_exploiting_x = torch.where(X[:,0] > 100)
        #    X_e = X[where_exploiting_x]

        current_time = times[:,-1]
        mask_per_path,rates = rate_function(X, current_time)

        # Time to next reactions
        rates_sum = torch.sum(rates, axis=1)

        # corrects rate in case 0 probabilities occur
        where_no_events = torch.where(rates_sum == 0.)
        rates[where_no_events] = 1.
        rates_sum = torch.sum(rates, axis=1)
        transition_probabilities = rates/rates_sum[:,None]

        time_between_events = Exponential(rates_sum).sample()
        new_times = current_time + time_between_events

        # selects next state
        which_state_to_take = Categorical(transition_probabilities).sample()
        mask_to_apply = choose_new_states(mask_per_path,which_state_to_take)
        new_states = X.clone() + mask_to_apply

        #makes sure that at no probabilities stays the same
        new_states[where_no_events] = X[where_no_events]

        #update paths and times
        paths = torch.concatenate([paths,
                                 new_states.unsqueeze(1)], dim=1)

        times = torch.concatenate([times,
                                   new_times.unsqueeze(1)],dim=1)
    
    return paths,times

def gillespie(X0,Q,number_of_times=100)->Tuple[torch.Tensor,torch.Tensor]:
    """
    # http://be150.caltech.edu/2019/handouts/12_stochastic_simulation_all_code.html

    parameters
    ----------

    X0: initial conditions
    q_rate_function: function -> new_states_available,rates

    returns
    -------
    paths,times
    """
    num_states = Q.size(0)
    batch_size = X0.shape[0]

    rates_from_Q  = lambda X,Q : Q[X]
    Q_r = Q.clone()
    Q_r[range(num_states),range(num_states)] = 0

    # Initialize process
    times = torch.full((batch_size, 1), 0.)
    paths = X0.unsqueeze(1)

    for time_index in range(number_of_times):
        X = paths[:,-1]
        current_time = times[:,-1]

        # rates and probabilities
        rates = rates_from_Q(X, Q_r)
        rates_sum = -Q[X,X].squeeze() #diagonal sum to 0
        transition_probabilities = rates/rates_sum[:,None]

        #times
        time_between_events = Exponential(rates_sum).sample()
        new_times = current_time + time_between_events

        # selects next state
        new_states = Categorical(transition_probabilities).sample()

        #update paths and times
        paths = torch.concatenate([paths,
                                   new_states.unsqueeze(1)], dim=1)

        times = torch.concatenate([times,
                                   new_times.unsqueeze(1)],dim=1)
    
    return paths,times

if __name__=="__main__":
    from matplotlib import pyplot as plt

    """
    lv_parameters = LotkaVolterraParameters()
    X0 = torch.Tensor([[19.,7.]]).repeat((10,1))

    new_states,rates = lotka_volterra_rate(X0,lv_parameters)
    rate_function = lambda x,t: lotka_volterra_rate(x,lv_parameters)

    paths,times = gillespie_mask(X0,rate_function,number_of_times=300)

    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,3))
    ax = axs[1]
    ax.set_title("Prey X")
    for paths_index in range(paths.size(0)):
        ax.plot(times[paths_index],paths[paths_index,:,0])

    ax = axs[0]
    ax.set_title("Predator X")
    for paths_index in range(paths.size(0)):
        ax.plot(times[paths_index],paths[paths_index,:,1])
    
    plt.show()
    """
    """
    r_parameters = RepressilatorParameters()
    X0 = np.array([10, 10, 10, 10, 10, 10, 0, 0, 0], dtype=float)
    X0 = torch.tensor(X0)[None,:].repeat((10,1))
    rate_function = lambda x,t: repressilator_rate(x,r_parameters)

    paths,times = gillespie_mask(X0,rate_function,number_of_times=10000)

    path_index = 0
    plt.plot(times[path_index],paths[path_index,:,1])
    plt.show()
    """
    d_param = DirichletPriorOnRatesParam()
    Q = generate_transition_rate_matrix(d_param)
    X0 = torch.randint(1,d_param.num_states,(3,))
    paths,times = gillespie(X0,Q,number_of_times=100)
    print(paths.shape)


