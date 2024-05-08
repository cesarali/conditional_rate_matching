import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def histogram_per_dimension_plot(states_histogram_at_0,
                                 states_histogram_at_1,
                                 paths_histogram,
                                 time_,
                                 states_legends=None,
                                 max_number_of_states_displayed=8,
                                 save_path=None):
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
    if isinstance(paths_histogram,torch.Tensor):
        paths_histogram = paths_histogram.detach().cpu()
    if isinstance(states_histogram_at_0,torch.Tensor):
        states_histogram_at_0 = states_histogram_at_0.detach().cpu()
    if isinstance(states_histogram_at_1,torch.Tensor):
        states_histogram_at_1 = states_histogram_at_1.detach().cpu()

    start_target = paths_histogram[0, :]
    end_target = paths_histogram[-1, :]

    number_of_dimensions = states_histogram_at_0.size(0)
    number_of_total_states = states_histogram_at_0.size(1)
    if states_legends is None:
        states_legends = [str(a) for a in range(number_of_total_states)]

    # create the layout with GridSpec

    # Create a GridSpec object
    fig, axs = plt.subplots(figsize=(12, 6))
    outer_ax = fig.axes[0]
    outer_ax.set_axis_off()

    gs = GridSpec(nrows=number_of_dimensions, ncols=3,
                  width_ratios=[1, 5, 1],
                  hspace=.6,
                  left=0.05, right=0.95, bottom=0.1, top=0.9)  # Adjust hspace for vertical spacing


    #fig.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.8, hspace=0.1)

    for dimension_index in range(number_of_dimensions):
        ax1 = fig.add_subplot(gs[dimension_index, 0])
        ax2 = fig.add_subplot(gs[dimension_index, 1])
        ax3 = fig.add_subplot(gs[dimension_index, 2])

        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                        labelleft=False)
        ax2.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                        labelleft=False)
        ax3.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
                        labelleft=False)


        ax1.set_title(r"$P_0(x)$")
        ax2.set_title("Rates per dimensions")
        ax3.set_title(r"$P_1(x)$")

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for spin_state_index in range(min(max_number_of_states_displayed,number_of_total_states)):
            state_spins_vector = states_legends[spin_state_index]
            state_label = "{0}:".format(spin_state_index) + " " + str(state_spins_vector)
            color = colors[spin_state_index % len(colors)]
            # create the main plot
            ax2.plot(time_, paths_histogram[:,dimension_index, spin_state_index], "*", alpha=0.4, color=color)

        #ax2.legend(loc='upper center',bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=4)


        ax1.bar(range(number_of_total_states), states_histogram_at_0[dimension_index,:].tolist(),
                alpha=0.3,label="Data 0 ",color=colors[0])
        ax1.bar(range(number_of_total_states), start_target[dimension_index,:].tolist(),
                alpha=0.3,label="Backward",color=colors[1])
        #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        ax3.bar(range(number_of_total_states), states_histogram_at_1[dimension_index,:].tolist(),alpha=0.3,
                label="Target T",color=colors[0])
        ax3.bar(range(number_of_total_states), end_target[dimension_index,:].tolist(), alpha=0.3,
                label="Forward",color=colors[1])
        #ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=1)

        ax1.set_ylim(0.,1.)
        ax3.set_ylim(0.,1.)

    # Remove ticks from the figure
    #plt.tick_params(axis='both', which='both', bottom=False, top=False,
    #                labelbottom=False, right=False, left=False, labelleft=False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def histograms_per_time_step(histograms,histograms_2,time_grid,save_path=None):
    """

    :param histograms:
    :param histograms_2:
    :param time_grid:
    :param save_path:
    :return:
    """
    if histograms_2 is not None:
        assert histograms_2.size(0) == histograms.size(0)

    bin_edges = range(histograms.size(1))
    fig, axs = plt.subplots(figsize=(16, 2),ncols=len(histograms))

    for histogram_index in range(histograms.size(0)):
        histogram = histograms[histogram_index]
        time_ = time_grid[histogram_index]
        ax1 = axs[histogram_index]
        ax1.set_title(f"Time {round(time_.item(),2)}")
        ax1.bar(bin_edges, histogram.detach().cpu().numpy(), align='edge', width=1.0, alpha=0.2, label="generated_0")
        if histograms_2 is not None:
            histogram2 = histograms_2[histogram_index]
            ax1.bar(bin_edges, histogram2.detach().cpu().numpy(), align='edge', width=1.0, alpha=0.2, label="generated_0")
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,labelleft=False)
        ax1.set_ylim(0.,1.)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def rates_plot(states_histogram_at_0,
               states_histogram_at_1,
               rates_histogram,
               time_,
               save_path=None,
               title="Average Rate Per Dimension",
               log_scale=False):
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
    states_histogram_at_0 = states_histogram_at_0.cpu().numpy()
    states_histogram_at_1 = states_histogram_at_1.cpu().numpy()
    rates_histogram = rates_histogram.cpu().numpy()
    time_ = time_.cpu().numpy()

    number_of_total_states = states_histogram_at_0.shape[0]
    # create the layout with GridSpec
    fig, axs = plt.subplots(figsize=(12, 3),
                            nrows=1, ncols=3,
                            gridspec_kw={'width_ratios': [1, 5, 1]},)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.8, hspace=0.1)

    ax1 = axs[0]
    ax1.set_title(r"$P^d_0(x)$")
    ax2 = axs[1]
    ax2.set_title(title)
    ax3 = axs[2]
    ax3.set_title(r"$P^d_1(x)$")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for spin_state_index in range(number_of_total_states):
        color = colors[0]
        # create the main plot
        ax2.plot(time_, rates_histogram[:, spin_state_index], "-", alpha=0.2, color="r")
    if log_scale:
        ax2.set_yscale('log')
    ax1.bar(range(number_of_total_states), states_histogram_at_0.tolist(),alpha=0.4,color=colors[0])
    ax3.bar(range(number_of_total_states), states_histogram_at_1.tolist(),alpha=0.4,color=colors[0])

    ax1.set_ylim(0.,1.)
    ax3.set_ylim(0.,1.)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)