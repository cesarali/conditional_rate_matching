"""
Here we create complementary datasets for the datasets such that they posses characteristics friendly to
the bridges

"""
import os
import torch
import numpy as np
import networkx as nx
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
def estimate_alpha(degrees, xmin):
    """
    Estimate the alpha parameter of a power-law distribution using MLE.
    :param degrees: list of observed degrees
    :param xmin: minimum value of degree to consider
    :return: estimated alpha
    """
    degrees = np.array(degrees)
    degrees = degrees[degrees >= xmin]
    n = len(degrees)
    alpha = 1 + n / np.sum(np.log(degrees / xmin))
    return alpha


def sample_power_law(alpha, xmin, size):
    """
    Sample degrees from a power-law distribution.
    :param alpha: power-law exponent
    :param xmin: minimum value of degree
    :param size: number of samples
    :return: sampled degrees
    """
    r = np.random.uniform(0, 1, size)
    samples = xmin * (1 - r) ** (-1 / (alpha - 1))
    return np.round(samples).astype(int)


def obtain_power_law_graph(networkx_graph):
    """
    """
    degrees = [d for n, d in networkx_graph.degree() if d != 0]
    number_of_missing_nodes = networkx_graph.number_of_nodes() - len(degrees)

    # Estimate alpha using MLE
    xmin = min(degrees)
    alpha = estimate_alpha(degrees, xmin)

    # Sample a new degree sequence
    sampled_degrees = sample_power_law(alpha, xmin, len(degrees))

    # correct degree sample
    if not sum(sampled_degrees) % 2 == 0:
        sampled_degrees[0] = sampled_degrees[0] + 1

        # Generate a new graph using the configuration model
    new_graph = nx.configuration_model(sampled_degrees)
    new_graph = nx.Graph(new_graph)  # Remove parallel edges and self-loops
    new_graph.remove_edges_from(nx.selfloop_edges(new_graph))
    for i in range(number_of_missing_nodes):
        new_graph.add_node(len(new_graph) + i)
    return new_graph, alpha

def obtain_graph_dataset(dataloader,obtain_graph_equivalent=obtain_power_law_graph):
    if isinstance(dataloader,GraphDataloaders):
        print("Creating Power Law Data Set For Bridge")

        train_graphs = []
        for databatch in dataloader.train():
            networkx_batch = dataloader.sample_to_graph(databatch[0])
            for networkx_graph in networkx_batch:
                new_graph, alpha = obtain_graph_equivalent(networkx_graph)
                train_graphs.append(new_graph)

        test_graphs = []
        for databatch in dataloader.test():
            networkx_batch = dataloader.sample_to_graph(databatch[0])
            for networkx_graph in networkx_batch:
                new_graph, alpha = obtain_graph_equivalent(networkx_graph)
                test_graphs.append(new_graph)

        return train_graphs,test_graphs


