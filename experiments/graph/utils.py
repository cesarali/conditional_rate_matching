import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching import results_path
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping
from conditional_rate_matching.models.metrics.graphs_metrics import eval_graph_list
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph import experiment_comunity_small

from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (ConstantThermostatConfig, 
                                                                                         PeriodicThermostatConfig,
                                                                                         ExponentialThermostatConfig,
                                                                                         PolynomialThermostatConfig,
                                                                                         PlateauThermostatConfig
                                                                                         )
def generate_graph_samples(path,
                           num_timesteps=100,
                           time_epsilon=0.005, 
                           device="cpu"):

    crm = CRM(experiment_dir=path, device=device)
    crm.config.pipeline.time_epsilon = time_epsilon
    crm.config.pipeline.num_intermediates = num_timesteps
    crm.config.pipeline.number_of_steps = num_timesteps
    source = crm.dataloader_0.test()
    target = crm.dataloader_1.test()

    x_1, x_0, x_test = [], [], []

    for batch in target:
        test_images = batch[0]
        x_test.append(test_images)
    
    for batch in source:
        input_graphs = batch[0]
        gen_graphs = crm.pipeline(sample_size=input_graphs.shape[0], return_intermediaries=False, train=False, x_0=input_graphs.to(crm.device))
        x_0.append(input_graphs)
        x_1.append(gen_graphs.detach().cpu())
    
    x_0 = to_adjecency_matrices(torch.cat(x_0), symmetrized=True if 'mlp' in path else False)
    x_1 = to_adjecency_matrices(torch.cat(x_1), symmetrized=True if 'mlp' in path else False)
    x_test = to_adjecency_matrices(torch.cat(x_test), symmetrized=True) if 'mlp' in path else torch.cat(x_test)
      
    torch.save(x_0, os.path.join(path, "sample_gen_x0.dat"))      
    torch.save(x_1, os.path.join(path, "sample_gen_x1.dat"))      
    torch.save(x_test, os.path.join(path, "sample_gen_test.dat"))
    
    return x_0, x_1, x_test


#################################################################

def graph_grid(sample, save_path='.', nrow=1, node_size=10, edge_width=2, node_color="darkred", edge_color="gray", figsize=(4, 4)):
    num_img = sample.shape[0]    
    ncol = math.ceil(num_img / nrow)
    _, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for i in range(num_img):
        adj_matrix = sample[i].numpy()
        G = nx.from_numpy_array(adj_matrix)
        singleton_nodes = list(nx.isolates(G))
        G.remove_nodes_from(singleton_nodes)
        pos = nx.spring_layout(G, seed=1234)
        nx.draw(G, pos, ax=axes[i], node_size=node_size, width=edge_width, node_color=node_color, edge_color=edge_color, with_labels=False)

    for j in range(num_img, nrow * ncol):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path+'/selected_sample.png')
    plt.show()

#################################################################

def graph_conditional_bridge(source, 
                             target, 
                             thermostat="constant", 
                             thermostat_params=(.1,0),
                             figsize=None, 
                             num_timesteps=100,
                             num_timesteps_displayed=10,
                             add_time_title=False,
                             save_path=None):


    config = experiment_comunity_small(network="simple")
    
    if thermostat == 'exponential':
        config.thermostat = ExponentialThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
        config.thermostat.max = thermostat_params[1]
    elif thermostat == 'constant':
        config.thermostat = ConstantThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
    elif thermostat == 'periodic':
        config.thermostat = PeriodicThermostatConfig()
        config.thermostat.gamma= thermostat_params[0]
        config.thermostat.max = thermostat_params[1]
    elif thermostat == 'polynomial':
        config.thermostat = PolynomialThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
        config.thermostat.exponent = thermostat_params[1]
    elif thermostat == 'plateau':
        config.thermostat = PlateauThermostatConfig()
        config.thermostat.gamma = thermostat_params[0]
        config.thermostat.slope = thermostat_params[1]
        config.thermostat.shift = thermostat_params[2]
        
    configure_thermostat(config, thermostat, thermostat_params)
    
    crm = CRM(config)
    crm.config.pipeline.number_of_steps = num_timesteps
    crm.config.pipeline.num_intermediates = num_timesteps_displayed

    num_grph, N, _ = source.shape
    print("dim adj = ", N)

    rate_model = lambda x, t: crm.forward_rate.conditional_transition_rate(x, target.view(-1,  N * N), t)
    gph_1, gph_hist, _ , time = TauLeaping(crm.config, rate_model, source.view(-1, N * N), forward=True)
    gph_1 = gph_1.long().view(-1,  N, N)
    gph_hist = gph_hist.long().view(-1, gph_hist.shape[1], N, N)
    path = torch.cat([gph_hist , gph_1.unsqueeze(1)], dim=1)

    nrow = num_grph
    ncol = path.shape[1]

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)
    time = torch.cat([time, torch.tensor([1.0])])

    for i in range(nrow):
        for j in range(ncol):
            adj_matrix = path[i][j].numpy()
            G = nx.from_numpy_array(adj_matrix)
            singleton_nodes = list(nx.isolates(G))
            G.remove_nodes_from(singleton_nodes)
            pos = nx.spring_layout(G, seed=1234)
            nx.draw(G, pos, ax=axes[i,j], node_size=2, width=0.3, node_color="darkred", edge_color="gray", with_labels=False)
            
            if i == 0 and add_time_title:
                axes[i,j].set_title(f"t={time[j]:.2f}", fontsize=8)
    
    # for ax in axes:
    #     ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/combined_path.png")
    plt.show()

    return gph_1, gph_hist, time


# def graph_conditional_bridge(source, 
#                              target, 
#                              thermostat="constant", 
#                              thermostat_params=(.1,0),
#                              figsize=None, 
#                              num_timesteps=100,
#                              num_timesteps_displayed=10,
#                              save_path=None):
    
#     config = experiment_comunity_small(network="simple")
    
#     if thermostat == 'exponential':
#         config.thermostat = ExponentialThermostatConfig()
#         config.thermostat.gamma = thermostat_params[0]
#         config.thermostat.max = thermostat_params[1]
#     elif thermostat == 'constant':
#         config.thermostat = ConstantThermostatConfig()
#         config.thermostat.gamma = thermostat_params[0]
#     elif thermostat == 'periodic':
#         config.thermostat = PeriodicThermostatConfig()
#         config.thermostat.gamma= thermostat_params[0]
#         config.thermostat.max = thermostat_params[1]
#     elif thermostat == 'polynomial':
#         config.thermostat = PolynomialThermostatConfig()
#         config.thermostat.gamma = thermostat_params[0]
#         config.thermostat.exponent = thermostat_params[1]
#     elif thermostat == 'plateau':
#         config.thermostat = PlateauThermostatConfig()
#         config.thermostat.gamma = thermostat_params[0]
#         config.thermostat.slope = thermostat_params[1]
#         config.thermostat.shift = thermostat_params[2]

#     crm = CRM(config)
#     crm.config.pipeline.number_of_steps = num_timesteps
#     crm.config.pipeline.num_intermediates = num_timesteps_displayed

#     num_grph, N, _ = source.shape
#     print("dim adj = ", N)

#     rate_model = lambda x, t: crm.forward_rate.conditional_transition_rate(x, target.view(-1,  N * N), t)
#     gph_1, gph_hist, _ , time = TauLeaping(crm.config, rate_model, source.view(-1, N * N), forward=True)
#     gph_1 = gph_1.long().view(-1,  N, N)
#     gph_hist = gph_hist.long().view(-1, gph_hist.shape[1], N, N)
#     path = torch.cat( [gph_hist , gph_1.unsqueeze(1)], dim=1)

#     for i in range(num_grph):
#         graph_grid(path[i], nrow=1, node_size=2, edge_width=0.3, figsize=figsize)

#     return gph_1, gph_hist, time

#################################################################

def graph_metrics(path, num_ensembles=1, num_timesteps=100, time_epsilon=0.005, device="cpu"):

    degree, cluster, orbit = [], [], []
    metric_ens = {}

    if num_ensembles > 1:
        print(f"INFO: Computing graph metrics with {num_ensembles} ensembles")

    for _ in range(num_ensembles):
        _, sample, test = generate_graph_samples(path, num_timesteps, time_epsilon, device)
        graphs_sample = []
        graphs_test = []

        for adj in sample:
            adj = adj.detach().cpu().numpy()
            g = nx.from_numpy_array(adj)
            graphs_sample.append(g)

        for adj in test:
            adj = adj.detach().cpu().numpy()
            g = nx.from_numpy_array(adj)
            graphs_test.append(g)

        metrics = eval_graph_list(graphs_test,
                                  graphs_sample,
                                  windows=False,
                                  orca_dir="/home/df630/conditional_rate_matching/src/conditional_rate_matching/models/metrics/orca_new_jersey")
        
        degree.append(metrics['degree'])
        cluster.append(metrics['cluster'])
        orbit.append(metrics['orbit'])

    metric_ens['degree'] = np.array(degree).mean()
    metric_ens['cluster'] = np.array(cluster).mean()
    metric_ens['orbit'] = np.array(orbit).mean() 
    
    return metric_ens




##### aux. functions




def to_adjecency_matrices(adjacency_tensor, symmetrized=False):

    N, M = adjacency_tensor.shape  
    D = int((1 + (1 + 8 * M)**0.5) / 2) if symmetrized else int(np.sqrt(M))
    A = torch.zeros((N, D, D))
    
    for idx in range(N):
        U = torch.zeros((D, D)) # upper triangular matrix
        U[torch.triu(torch.ones(D, D), diagonal=1).bool()] = adjacency_tensor[idx]
        A[idx] = U + U.T
    return A



def generate_samples(path, 
                     x0_input,
                     num_timesteps=100, 
                     time_epsilon=0.0,
                     device="cpu"):
    
    crm = CRM(experiment_dir=path, device=device)
    crm.config.pipeline.time_epsilon = time_epsilon
    crm.config.pipeline.num_intermediates = num_timesteps
    crm.config.pipeline.number_of_steps = num_timesteps

    x_1, x_t, t = crm.pipeline(x_test.shape[0], 
                               return_intermediaries=True, 
                               train=False, 
                               x_0=x0_input)
    
    return x_1, x_t, t


def configure_thermostat(config, thermostat, params):
    thermostat_config_classes = {'exponential': ExponentialThermostatConfig,
                                 'constant': ConstantThermostatConfig,
                                 'periodic': PeriodicThermostatConfig,
                                 'polynomial': PolynomialThermostatConfig,
                                 'plateau': PlateauThermostatConfig
                                 }
    
    thermostat_config = thermostat_config_classes[thermostat]()
    gamma, *other_params = params
    thermostat_config.gamma = gamma

    if hasattr(thermostat_config, 'max'):
        thermostat_config.max = other_params[0]
    if hasattr(thermostat_config, 'exponent'):
        thermostat_config.exponent = other_params[1]
    if hasattr(thermostat_config, 'slope'):
        thermostat_config.slope = other_params[1]
    if hasattr(thermostat_config, 'shift'):
        thermostat_config.shift = other_params[2]
    config.thermostat = thermostat_config
