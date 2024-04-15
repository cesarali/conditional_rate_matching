import os
import numpy as np
from matplotlib import pyplot as plt
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

def music_plot_conditional(generative_sample,origin_sample,config:CRMConfig,plot_path):
    generation_dimension = config.data1.generation_dimension
    conditional_dimension = config.data1.conditional_dimension

    idx = 0
    plt.scatter(np.arange(256), generative_sample[idx, :], alpha=0.5,label="generation")
    plt.scatter(np.arange(256), origin_sample[idx, :], alpha=0.5,label="original")
    plt.legend(loc="best")
    if plot_path is not None:
        plt.savefig(plot_path)

    print(f"generation dimension {generation_dimension}")
    print(plot_path)
    print(generative_sample.shape)
    print(origin_sample.shape)
