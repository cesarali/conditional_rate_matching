import torch
from torch.nn import functional as F
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

def hellinger_distance_function(hist1, hist2):
    """
    Compute the Hellinger distance between two histograms.

    Args:
    hist1 (torch.Tensor): Histogram of the first distribution.
    hist2 (torch.Tensor): Histogram of the second distribution.

    Returns:
    float: Hellinger distance between the two histograms.
    """
    # Normalize histograms to make them probability distributions
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Compute the square root of each bin
    sqrt_hist1 = torch.sqrt(hist1)
    sqrt_hist2 = torch.sqrt(hist2)

    # Compute the Euclidean distance between the square-rooted histograms
    euclidean_distance = torch.norm(sqrt_hist1 - sqrt_hist2, 2)

    # Normalize to get the Hellinger distance
    hellinger_distance = euclidean_distance / torch.sqrt(torch.tensor(2.0))

    return hellinger_distance


def hellinger_distance(generative,original,config:CRMConfig):
    conditional_dimension = config.data1.conditional_dimension
    vocab_size = config.data1.vocab_size

    generative_sample_complete_part = generative[:,conditional_dimension:]
    original_complete_part = original[:,conditional_dimension:]
    generative_dim = original_complete_part.size(1)
    batch_size = original_complete_part.size(0)

    generative_sample_complete_part = generative_sample_complete_part.reshape(batch_size*generative_dim)
    original_complete_part = original_complete_part.reshape(batch_size*generative_dim)

    generative_histogram = F.one_hot(generative_sample_complete_part.long(), num_classes=vocab_size)
    original_histogram = F.one_hot(original_complete_part.long(),  num_classes=vocab_size)
    original_histogram = original_histogram.sum(axis=0)
    generative_histogram = generative_histogram.sum(axis=0)

    return hellinger_distance_function(generative_histogram,original_histogram)