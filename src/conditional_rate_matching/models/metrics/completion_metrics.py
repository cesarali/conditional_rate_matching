import torch
from torch.nn import functional as F
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
import numpy as np

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

    return hellinger_distance_function(generative_histogram,original_histogram).item()


def outlier_per_song(generated_song_notes, real_song_notes,config):
    notes_in_generated_song = set(generated_song_notes.detach().numpy())
    notes_in_real_song = set(real_song_notes.detach().numpy())

    notes_in_generated_not_in_real = notes_in_generated_song - notes_in_real_song
    number_of_notes_not = len(notes_in_generated_not_in_real)

    total_proportion = number_of_notes_not / config.data1.vocab_size

    tensor_notes_in_real = torch.Tensor(list(notes_in_real_song))
    # Create a boolean mask for each element in tensor_a indicating whether it's in tensor_b
    mask = torch.isin(generated_song_notes, tensor_notes_in_real)

    # Count the number of True values in the mask
    count_of_notes_in_real = torch.sum(mask).item()
    count_not_in_real = real_song_notes.size(0) - count_of_notes_in_real

    outliers_per_song = count_not_in_real / real_song_notes.size(0)

    return total_proportion, outliers_per_song


def outliers(generative,original,config:CRMConfig):
    number_of_songs = generative.size(0)

    total_proportion_list = []
    outlier_per_song_list = []

    for song_index in range(number_of_songs):
        generated_song_notes = generative[song_index]
        real_song_notes = original[song_index]
        proportion, outlier_ = outlier_per_song(generated_song_notes, real_song_notes,config)
        total_proportion_list.append(proportion)
        outlier_per_song_list.append(outlier_)

    total_proportion_list = np.asarray(total_proportion_list)
    outlier_per_song_list = np.asarray(outlier_per_song_list)

    total_proportion_mean, total_proportionl_std = total_proportion_list.mean(), total_proportion_list.std()
    outlier_per_song_mean, outlier_per_song_std = outlier_per_song_list.mean(), outlier_per_song_list.std()

    return {"total_proportion_mean":total_proportion_mean,
            "total_proportionl_std":total_proportionl_std,
            "outlier_per_song_mean":outlier_per_song_mean,
            "outlier_per_song_std":outlier_per_song_std}