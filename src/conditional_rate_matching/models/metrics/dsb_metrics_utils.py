import json
from dataclasses import dataclass
from typing import Union
from matplotlib import pyplot as plt
from conditional_rate_matching.models.generative_models.dsb import DSB

from conditional_rate_matching.models.metrics.distances import kmmd,marginal_histograms
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_categorical_histogram_per_dimension

from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader_iterator
from conditional_rate_matching.utils.plots.paths_plots import histograms_per_time_step

import torch.nn.functional as F

from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.models.metrics.crm_path_metrics import classification_path
from conditional_rate_matching.data.gray_codes_dataloaders import GrayCodeDataLoader

#binary
from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_marginals_binary_histograms
from conditional_rate_matching.models.metrics.metrics_utils import store_metrics
from conditional_rate_matching.models.metrics.dsb_metrics import sb_plot

key_in_dict = lambda dictionary, key: dictionary is not None and key in dictionary

def sample_dsb(dsb,current_model,sinkhorn_iteration):
    start_dataloader,is_past_forward = dsb.pipeline.direction_of_past_model(sinkhorn_iteration=sinkhorn_iteration)
    is_current_forward = not is_past_forward

    config = dsb.config
    vocab_size, dimensions, max_test_size = config.data0.vocab_size, config.data0.dimensions, config.trainer.max_test_size
    test_sample = sample_from_dataloader_iterator(start_dataloader.test(), sample_size=max_test_size).to(dsb.device)

    generative_sample, generative_path, ts = dsb.pipeline(sample_size=test_sample.shape[0],
                                                          model=current_model,
                                                          forward=is_current_forward,
                                                          return_intermediaries=True,
                                                          train=False)
    sizes = (vocab_size, dimensions, max_test_size)
    return sizes,generative_sample, generative_path,ts,test_sample

@dataclass
class DSBMetricsAvaliable:
    mse_histograms: str = "mse_histograms"
    marginal_binary_histograms: str = "marginal_binary_histograms"
    sb_plot:str = "sb_plot"

def log_dsb_metrics(generative_model:DSB, current_model, past_model, epoch, sinkhorn_iteration, all_metrics = {}, metrics_to_log=None, where_to_log=None, writer=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    config = generative_model.config
    if metrics_to_log is None:
        metrics_to_log = config.trainer.metrics

    #OBTAIN SAMPLES
    sizes,generative_sample, generative_path,ts,test_sample = sample_dsb(generative_model,current_model,
                                                                         sinkhorn_iteration=sinkhorn_iteration)
    vocab_size, dimensions, max_test_size = sizes
    size_ = min(generative_sample.size(0),test_sample.size(0))
    generative_sample = generative_sample[:size_]
    test_sample = test_sample[:size_]

    # HISTOGRAMS METRICS
    metric_string_name = "mse_histograms"
    if metric_string_name in metrics_to_log:
        mse_marginal_histograms = marginal_histograms(generative_sample,test_sample)
        mse_metrics = {"mse_marginal_histograms": mse_marginal_histograms.tolist()}
        all_metrics = store_metrics(generative_model.experiment_files, all_metrics, new_metrics=mse_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)

    # MARGINAL PLOTS
    metric_string_name = "marginal_binary_histograms"
    if metric_string_name in metrics_to_log:
        assert vocab_size == 2
        histograms_generative = generative_sample.mean(dim=0)

        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = generative_model.experiment_files.plot_path.format("marginal_binary_histograms_{0}".format(epoch))

        histogram0 = binary_histogram_dataloader(generative_model.dataloader_0, dimensions=dimensions,
                                                 train=True, maximum_test_sample_size=max_test_size)
        histogram1 = binary_histogram_dataloader(generative_model.dataloader_1, dimensions=dimensions,
                                                 train=True, maximum_test_sample_size=max_test_size)

        marginal_histograms_tuple = (histogram0, histogram0, histogram1, histograms_generative)
        plot_marginals_binary_histograms(marginal_histograms_tuple,plots_path=plot_path)

    # SB PLOT
    metric_string_name = "sb_plot"
    if metric_string_name in metrics_to_log:
        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = generative_model.experiment_files.plot_path.format("bridge_plot_{0}".format(epoch))

        sb_plot(generative_model,current_model,past_model,sinkhorn_iteration,
                save_path=plot_path,max_number_of_states_displayed = 8)

    return all_metrics

