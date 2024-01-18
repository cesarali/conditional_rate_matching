import json
from dataclasses import dataclass
from typing import Union
from matplotlib import pyplot as plt
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.generative_models.ctdd import CTDD
from conditional_rate_matching.models.generative_models.oops import Oops

from conditional_rate_matching.models.metrics.distances import kmmd,marginal_histograms
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_categorical_histogram_per_dimension

from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader
from conditional_rate_matching.utils.plots.paths_plots import histograms_per_time_step

import torch.nn.functional as F

from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.models.metrics.crm_path_metrics import classification_path
from conditional_rate_matching.data.gray_codes_dataloaders import GrayCodeDataLoader

#binary
from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_marginals_binary_histograms

#mnist
from conditional_rate_matching.utils.plots.images_plots import mnist_grid
from conditional_rate_matching.models.metrics.fid_metrics import fid_nist
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

#graph
from conditional_rate_matching.models.metrics.graphs_metrics import eval_graph_list
from conditional_rate_matching.utils.plots.graph_plots import plot_graphs_list2
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.utils.plots.gray_code_plots import plot_samples

key_in_dict = lambda dictionary, key: dictionary is not None and key in dictionary

@dataclass
class MetricsAvaliable:
    mse_histograms: str = "mse_histograms"
    kdmm: str = "kdmm"
    graphs_metrics: str = "graphs_metrics"
    graphs_plot: str = "graphs_plot"
    categorical_histograms: str = "categorical_histograms"
    binary_paths_histograms: str = "binary_paths_histograms"
    marginal_binary_histograms: str = "marginal_binary_histograms"
    mnist_plot: str = "mnist_plot"
    fid_nist: str = "fid_nist"
    grayscale_plot: str = "grayscale_plot"

def store_metrics(experiment_files,all_metrics,new_metrics,metric_string_name,epoch,where_to_log=None):
    if key_in_dict(where_to_log, metric_string_name):
        mse_metric_path = where_to_log[metric_string_name]
    else:
        mse_metric_path = experiment_files.metrics_file.format(metric_string_name + "_{0}_".format(epoch))
    all_metrics.update(new_metrics)
    with open(mse_metric_path, "w") as f:
        json.dump(new_metrics, f)
    return all_metrics

def sample_crm(crm,config):
    source_dataloader = crm.dataloader_0
    data_dataloader = crm.dataloader_1
    vocab_size, dimensions, max_test_size = config.data1.vocab_size, config.data1.dimensions, config.trainer.max_test_size
    dataloader = crm.dataloader_1.test()
    test_sample = sample_from_dataloader(dataloader, sample_size=max_test_size).to(crm.device)
    generative_sample, generative_path, ts = crm.pipeline(sample_size=test_sample.shape[0],return_intermediaries=True,train=False)
    sizes = (vocab_size, dimensions, max_test_size)
    return sizes, source_dataloader,data_dataloader,generative_sample, generative_path, ts,test_sample

def sample_ctdd(ctdd,config):
    source_dataloader = ctdd.dataloader_1
    data_dataloader = ctdd.dataloader_0
    vocab_size, dimensions, max_test_size = config.data0.vocab_size, config.data0.dimensions, config.trainer.max_test_size
    dataloader = ctdd.dataloader_0.test()
    test_sample = sample_from_dataloader(dataloader, sample_size=max_test_size).to(ctdd.device)
    generative_sample = ctdd.pipeline(ctdd.backward_rate, sample_size=test_sample.shape[0],
                                                  device=ctdd.device)
    sizes = (vocab_size, dimensions, max_test_size)
    return sizes, source_dataloader, data_dataloader, generative_sample,test_sample

def sample_oops(oops,config):
    source_dataloader = None
    data_dataloader = oops.dataloader_0.test
    vocab_size, dimensions, max_test_size = config.data0.vocab_size, config.data0.dimensions, config.trainer.max_test_size
    dataloader = oops.dataloader_0.test()
    test_sample = sample_from_dataloader(dataloader, sample_size=max_test_size).to(oops.device)
    generative_sample, ll = oops.pipeline(oops.model, max_test_size, return_path=False, get_ll=True)
    sizes = (vocab_size, dimensions, max_test_size)
    return sizes,source_dataloader, data_dataloader, generative_sample,test_sample

def log_metrics(generative_model: Union[CRM,CTDD,Oops], epoch, all_metrics = {}, metrics_to_log=None, where_to_log=None, writer=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    config = generative_model.config
    if metrics_to_log is None:
        metrics_to_log = config.trainer.metrics

    #OBTAIN SAMPLES
    if isinstance(generative_model, CRM):
        sizes, source_dataloader,data_dataloader,generative_sample, generative_path,ts,test_sample = sample_crm(generative_model, config)
    elif isinstance(generative_model, CTDD):
        sizes, source_dataloader, data_dataloader, generative_sample,test_sample = sample_ctdd(generative_model,config)
    elif isinstance(generative_model,Oops):
        sizes,source_dataloader, data_dataloader, generative_sample,test_sample = sample_oops(generative_model,config)
    else:
        return {}

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

    metric_string_name = "kdmm"
    if metric_string_name in metrics_to_log:
        kdmm_ = kmmd(generative_sample,test_sample)
        kdmm_metrics = {"kdmm": kdmm_.item()}
        all_metrics = store_metrics(generative_model.experiment_files, all_metrics, new_metrics=kdmm_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)

    #FID's
    metric_string_name = "fid_nist"
    if metric_string_name in metrics_to_log:
        # Here calculates the fid score in the same device as the trainer
        if isinstance(config.data1, NISTLoaderConfig):
            mnist_config = config.data1
        else:
            mnist_config = config.data0
        fid_nist_metrics = fid_nist(generative_sample, test_sample,mnist_config.dataset_name,config.trainer.device)
        all_metrics = store_metrics(generative_model.experiment_files, all_metrics, new_metrics=fid_nist_metrics,
                                    metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)
        
    # GRAPHS
    if "graphs_metrics" in metrics_to_log or "graphs_plot" in metrics_to_log:
        if isinstance(data_dataloader,GraphDataloaders):
            test_graphs = data_dataloader.sample_to_graph(test_sample)
            generated_graphs = data_dataloader.sample_to_graph(generative_sample)

            metric_string_name = "graphs_metrics"
            if metric_string_name in metrics_to_log:
                try:
                    graph_metrics_ = eval_graph_list(test_graphs,
                                                     generated_graphs,
                                                     windows=config.trainer.berlin,
                                                     orca_dir=config.trainer.orca_dir)
                    all_metrics = store_metrics(generative_model.experiment_files, all_metrics, new_metrics=graph_metrics_, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)
                except:
                    print("Problem with graph metrics")

            metric_string_name = "graphs_plot"
            if metric_string_name in metrics_to_log:
                plot_path_generative = generative_model.experiment_files.plot_path.format("graphs_generative_{0}".format(epoch))
                plot_path = generative_model.experiment_files.plot_path.format("graphs_test")
                plot_graphs_list2(generated_graphs, title="Generative",save_dir=plot_path_generative)
                plot_graphs_list2(test_graphs, title="Test",save_dir=plot_path)

    #=======================================================
    #                          PLOTS
    #=======================================================

    # HISTOGRAMS PLOTS
    metric_string_name = "categorical_histograms"
    if metric_string_name in metrics_to_log:
        histogram0 = categorical_histogram_dataloader(source_dataloader, dimensions, vocab_size,
                                                      maximum_test_sample_size=max_test_size)
        histogram1 = categorical_histogram_dataloader(data_dataloader, dimensions, vocab_size,
                                                      maximum_test_sample_size=max_test_size)

        generative_histogram = F.one_hot(generative_sample.long(), vocab_size).sum(axis=0)
        generative_histogram = generative_histogram/generative_sample.size(0)
        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = generative_model.experiment_files.plot_path.format("categorical_histograms_{0}".format(epoch))
        plot_categorical_histogram_per_dimension(histogram0, histogram1, generative_histogram,save_path=plot_path, remove_ticks=False)

    metric_string_name = "binary_paths_histograms"
    if metric_string_name in metrics_to_log:
        if isinstance(generative_model, CRM):
            assert vocab_size == 2
            histograms_generative = generative_path.mean(axis=0)
            if key_in_dict(where_to_log,metric_string_name):
                plot_path = where_to_log[metric_string_name]
            else:
                plot_path = generative_model.experiment_files.plot_path.format("binary_path_histograms_{0}".format(epoch))
            rate_logits = classification_path(generative_model.forward_rate, test_sample, ts, )
            rate_probabilities = F.softmax(rate_logits, dim=2)[:,:,1]
            histograms_per_time_step(histograms_generative,rate_probabilities,ts,save_path=plot_path)

    metric_string_name = "marginal_binary_histograms"
    if metric_string_name in metrics_to_log:
        assert vocab_size == 2
        histograms_generative = generative_sample.mean(dim=0)

        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = generative_model.experiment_files.plot_path.format("marginal_binary_histograms_{0}".format(epoch))

        histogram0 = binary_histogram_dataloader(source_dataloader, dimensions=dimensions,
                                                 train=True, maximum_test_sample_size=max_test_size)
        histogram1 = binary_histogram_dataloader(data_dataloader, dimensions=dimensions,
                                                 train=True, maximum_test_sample_size=max_test_size)

        marginal_histograms_tuple = (histogram0, histogram0, histogram1, histograms_generative)
        plot_marginals_binary_histograms(marginal_histograms_tuple,plots_path=plot_path)

    #IMAGES PLOTS
    metric_string_name = "mnist_plot"
    if metric_string_name in metrics_to_log:
        assert vocab_size == 2
        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
            mnist_grid(generative_sample, plot_path)
        else:
            plot_path = generative_model.experiment_files.plot_path.format("nist_plot_{0}".format(epoch))
            plot_path_test = generative_model.experiment_files.plot_path.format("nist_plot_test_{0}".format(epoch))
            mnist_grid(generative_sample, plot_path)
            mnist_grid(test_sample, plot_path_test)

    #GRAYCODE PLOTS
    metric_string_name = MetricsAvaliable.grayscale_plot
    if metric_string_name in metrics_to_log:
        if isinstance(data_dataloader,GrayCodeDataLoader):
            plt.close()
            test_sample_gray_image = data_dataloader.get_images(test_sample)
            generative_sample_gray_image = data_dataloader.get_images(generative_sample)

            plot_path_test_gray = generative_model.experiment_files.plot_path.format("graycode_plot_test")
            plot_path_gray = generative_model.experiment_files.plot_path.format("graycode_plot_{0}".format(epoch))

            plot_samples(test_sample_gray_image, plot_path_test_gray , lim=data_dataloader.db.f_scale)
            plot_samples(generative_sample_gray_image,plot_path_gray,lim=data_dataloader.db.f_scale)

    return all_metrics