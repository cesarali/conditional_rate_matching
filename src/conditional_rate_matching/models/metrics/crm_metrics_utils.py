import torch
from typing import Union
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.generative_models.ctdd import CTDD
from conditional_rate_matching.models.metrics.metrics_utils import store_metrics
from conditional_rate_matching.models.metrics.distances import kmmd,marginal_histograms
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_categorical_histogram_per_dimension

from conditional_rate_matching.models.pipelines.samplers_utils import sample_from_dataloader
from conditional_rate_matching.utils.plots.paths_plots import histograms_per_time_step

import torch.nn.functional as F

from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.models.metrics.crm_path_metrics import classification_path

#binary
from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_marginals_binary_histograms

#mnist
from conditional_rate_matching.utils.plots.images_plots import mnist_grid

#graph
from conditional_rate_matching.models.metrics.graphs_metrics import eval_graph_list
from conditional_rate_matching.utils.plots.graph_plots import plot_graphs_list2

key_in_dict = lambda dictionary, key: dictionary is not None and key in dictionary


def log_metrics(crm: Union[CRM,CTDD],epoch, metrics_to_log=None, where_to_log=None, writer=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    all_metrics = {}

    config = crm.config
    if metrics_to_log is None:
        metrics_to_log = config.trainer.metrics

    #OBTAIN SAMPLES
    dataloader = crm.dataloader_1.test()

    test_sample = sample_from_dataloader(dataloader,sample_size=config.data1.max_test_size).to(crm.device)
    if isinstance(crm,CRM):
        generative_sample,generative_path,ts = crm.pipeline(sample_size=test_sample.shape[0],return_intermediaries=True,train=False)
    elif isinstance(crm,CTDD):
        generative_sample = crm.pipeline(crm.backward_rate, sample_size=test_sample.shape[0], device=crm.device)
    size_ = min(generative_sample.size(0),test_sample.size(0))

    generative_sample = generative_sample[:size_]
    test_sample = test_sample[:size_]

    # HISTOGRAMS METRICS
    metric_string_name = "mse_histograms"
    if metric_string_name in metrics_to_log:
        mse_marginal_histograms = marginal_histograms(generative_sample,test_sample)
        mse_metrics = {"mse_marginal_histograms": mse_marginal_histograms.tolist()}
        all_metrics = store_metrics(crm.experiment_files, all_metrics, new_metrics=mse_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)

    metric_string_name = "kdmm"
    if metric_string_name in metrics_to_log:
        mse_0 = kmmd(generative_sample,test_sample)
        mse_metrics = {"mse_histograms_0": mse_0.item()}
        all_metrics = store_metrics(crm.experiment_files, all_metrics, new_metrics=mse_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)

    # GRAPHS
    if "graphs_metrics" in metrics_to_log or "graphs_plot" in metrics_to_log:
        if isinstance(crm.dataloader_1,GraphDataloaders):
            test_graphs = crm.dataloader_1.sample_to_graph(test_sample)
            generated_graphs = crm.dataloader_1.sample_to_graph(generative_sample)

            metric_string_name = "graphs_metrics"
            if metric_string_name in metrics_to_log:
                try:
                    graph_metrics_ = eval_graph_list(test_graphs,generated_graphs,windows=config.trainer.berlin)
                    all_metrics = store_metrics(crm.experiment_files, all_metrics, new_metrics=graph_metrics_, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)
                except:
                    pass

            metric_string_name = "graphs_plot"
            if metric_string_name in metrics_to_log:
                plot_path_generative = crm.experiment_files.plot_path.format("graphs_generative_{0}".format(epoch))
                plot_path = crm.experiment_files.plot_path.format("graphs_test")
                plot_graphs_list2(generated_graphs, title="Generative",save_dir=plot_path_generative)
                plot_graphs_list2(test_graphs, title="Test",save_dir=plot_path)

    # HISTOGRAMS PLOTS
    metric_string_name = "categorical_histograms"
    if metric_string_name in metrics_to_log:
        histogram0 = categorical_histogram_dataloader(crm.dataloader_0, config.data1.dimensions, config.data1.vocab_size,
                                                      maximum_test_sample_size=config.data1.max_test_size)
        histogram1 = categorical_histogram_dataloader(crm.dataloader_1, config.data1.dimensions, config.data1.vocab_size,
                                                      maximum_test_sample_size=config.data1.max_test_size)

        generative_histogram = F.one_hot(generative_sample.long(), config.data1.vocab_size).sum(axis=0)
        generative_histogram = generative_histogram/generative_sample.size(0)
        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = crm.experiment_files.plot_path.format("categorical_histograms_{0}".format(epoch))
        plot_categorical_histogram_per_dimension(histogram0, histogram1, generative_histogram,save_path=plot_path, remove_ticks=False)

    metric_string_name = "binary_paths_histograms"
    if metric_string_name in metrics_to_log:
        if isinstance(crm, CRM):
            assert crm.config.data1.vocab_size == 2
            histograms_generative = generative_path.mean(axis=0)
            if key_in_dict(where_to_log,metric_string_name):
                plot_path = where_to_log[metric_string_name]
            else:
                plot_path = crm.experiment_files.plot_path.format("binary_path_histograms_{0}".format(epoch))
            rate_logits = classification_path(crm.forward_rate, test_sample, ts,)
            rate_probabilities = F.softmax(rate_logits, dim=2)[:,:,1]
            histograms_per_time_step(histograms_generative,rate_probabilities,ts,save_path=plot_path)

    metric_string_name = "marginal_binary_histograms"
    if metric_string_name in metrics_to_log:
        assert crm.config.data1.vocab_size == 2
        histograms_generative = generative_sample.mean(dim=0)

        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = crm.experiment_files.plot_path.format("marginal_binary_histograms_{0}".format(epoch))

        histogram0 = binary_histogram_dataloader(crm.dataloader_0, dimensions=config.data1.dimensions,
                                                 train=True, maximum_test_sample_size=config.data1.max_test_size)
        histogram1 = binary_histogram_dataloader(crm.dataloader_1, dimensions=config.data1.dimensions,
                                                 train=True, maximum_test_sample_size=config.data1.max_test_size)
        marginal_histograms_tuple = (histogram0, histogram0, histogram1, histograms_generative)
        plot_marginals_binary_histograms(marginal_histograms_tuple,plots_path=plot_path)

    #IMAGES PLOTS
    metric_string_name = "mnist_plot"
    if metric_string_name in metrics_to_log:
        assert crm.config.data1.vocab_size == 2
        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = crm.experiment_files.plot_path.format("mnist_plot_{0}".format(epoch))
            plot_path2 = crm.experiment_files.plot_path.format("mnist_plot_2_{0}".format(epoch))
        mnist_grid(generative_sample,plot_path)
        #mnist_grid(generative_path[:,-1,:],plot_path2)

    return all_metrics