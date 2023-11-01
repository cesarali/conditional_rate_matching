import os
import sys
import torch
import numpy as np
from typing import List,Tuple
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_categorical_histograms(config,histogram,t,save_path=None,show=True):
    K = config.number_of_states
    dimension = config.number_of_spins

    # Create a single figure with subplots for each dimension
    fig, axes = plt.subplots(1, dimension, figsize=(15, 3))

    for dim, ax in enumerate(axes):
        ax.bar(np.arange(K), histogram[dim, :].numpy())
        ax.set_title(f'Dimension {dim + 1}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_ylim([0, 1])  #
        ax.set_xticks(np.arange(K))

    plt.suptitle(f'Time {t}')
    plt.tight_layout()
    if save_path is None:
        if show:
            plt.show()
    return None


def plot_categorical_histogram_per_dimension(states_histogram_at_0,
                                             states_histogram_at_1,
                                             target_1,
                                             states_legends=None,
                                             save_path=None,
                                             remove_ticks=True):
        """
        Forward is the direction of the past model

        :param is_past_forward:
        :param time_:
        :param states_histogram_at_0:
        :param states_histogram_at_1:
        :param histogram_from_rate:
        :param states_legends:
        :return:
        """

        if isinstance(states_histogram_at_0, torch.Tensor):
            states_histogram_at_0 = states_histogram_at_0.detach().cpu()
        if isinstance(states_histogram_at_1, torch.Tensor):
            states_histogram_at_1 = states_histogram_at_1.detach().cpu()

        number_of_dimensions = states_histogram_at_0.size(0)
        number_of_total_states = states_histogram_at_0.size(1)
        if states_legends is None:
            states_legends = [str(a) for a in range(number_of_total_states)]

        # create the layout with GridSpec

        # Create a GridSpec object
        fig, axs = plt.subplots(figsize=(12, 6))
        outer_ax = fig.axes[0]
        outer_ax.set_axis_off()

        gs = GridSpec(nrows=number_of_dimensions, ncols=2,
                      width_ratios=[1, 1],
                      hspace=.6,
                      left=0.05, right=0.95, bottom=0.1, top=0.9)  # Adjust hspace for vertical spacing

        # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.8, hspace=0.1)

        for dimension_index in range(number_of_dimensions):
            ax1 = fig.add_subplot(gs[dimension_index, 0])
            ax3 = fig.add_subplot(gs[dimension_index, 1])

            if remove_ticks:
                ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                labelbottom=False,
                                labelleft=False)

                ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                                labelbottom=False,
                                labelleft=False)
            if dimension_index == 0:
                ax1.set_title(r"$P_0(x)$")
                ax3.set_title(r"$P_1(x)$")
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            # ax2.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)

            ax1.bar(range(number_of_total_states), states_histogram_at_0[dimension_index, :].tolist(),
                    alpha=0.3, label="Data 0 ", color=colors[0])
            #ax1.bar(range(number_of_total_states), start_target[dimension_index, :].tolist(),
            #        alpha=0.3, label="Backward", color=colors[1])
            # ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

            ax3.bar(range(number_of_total_states), states_histogram_at_1[dimension_index, :].tolist(), alpha=0.3,
                    label="Target T", color=colors[0])
            ax3.bar(range(number_of_total_states), target_1[dimension_index, :].tolist(), alpha=0.3,
                    label="Forward", color=colors[1])
            # ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

            ax1.set_ylim(0., 1.)
            ax3.set_ylim(0., 1.)

        # Remove ticks from the figure
        # plt.tick_params(axis='both', which='both', bottom=False, top=False,
        #                labelbottom=False, right=False, left=False, labelleft=False)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)


def plot_histograms(marginal_histograms:Tuple[torch.Tensor],plots_path=None):
    """

    :param marginal_histograms: List[] marginal_0,marginal_generated_0,marginal_1,marginal_noising_1
    :param plots_path:
    :return:
    """
    marginal_0,marginal_generated_0,marginal_1,marginal_noising_1 = marginal_histograms
    marginal_0,marginal_generated_0,marginal_1,marginal_noising_1 = marginal_0.numpy(),marginal_generated_0.numpy(),marginal_1.numpy(),marginal_noising_1.numpy()

    fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(12,4))

    bin_edges = range(len(marginal_0))
    ax1.bar(bin_edges, marginal_generated_0, align='edge', width=1.0,alpha=0.2,label="generated_0")
    ax1.bar(bin_edges, marginal_0, align='edge', width=1.0,alpha=0.2,label="data ")

    ax1.set_title(r'Time 0')
    ax1.set_xlabel('Bins')
    ax1.set_ylabel('Counts')
    ax1.legend(loc="upper right")

    ax2.bar(bin_edges,marginal_noising_1 , align='edge', width=1.0,alpha=0.2,label="generated_0")
    ax2.bar(bin_edges, marginal_1, align='edge', width=1.0,alpha=0.2,label="data ")

    ax2.set_title(r'Time 1')
    ax2.set_xlabel('Bins')
    ax2.set_ylabel('Counts')
    ax2.legend(loc="upper right")

    ax2.set_ylim(0, 1.)
    ax1.set_ylim(0, 1.)

    if plots_path is None:
        plt.show()
    else:
        plt.savefig(plots_path)

def kHistogramPlot(config,histogram,t,save_path=None,show=True):
    K = config.number_of_states
    dimension = config.number_of_spins

    # Create a single figure with subplots for each dimension
    fig, axes = plt.subplots(1, dimension, figsize=(15, 3))

    for dim, ax in enumerate(axes):
        ax.bar(np.arange(K), histogram[dim, :].numpy())
        ax.set_title(f'Dimension {dim + 1}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Frequency')
        ax.set_ylim([0, 1])  #
        ax.set_xticks(np.arange(K))

    plt.suptitle(f'Time {t}')
    plt.tight_layout()
    if save_path is None:
        if show:
            plt.show()