import torch
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.metrics.metrics_utils import store_metrics
from conditional_rate_matching.models.metrics.distances import kmmd,marginal_histograms
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_categorical_histogram_per_dimension

from conditional_rate_matching.models.pipelines.samplers_utils import sample_from_dataloader
from conditional_rate_matching.utils.plots.paths_plots import histograms_per_time_step

from conditional_rate_matching.models.metrics.crm_path_metrics import classification_path
import torch.nn.functional as F

key_in_dict = lambda dictionary, key: dictionary is not None and key in dictionary


def log_metrics(crm: CRM,epoch, metrics_to_log=None, where_to_log=None, writer=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    all_metrics = {}

    config = crm.config
    if metrics_to_log is None:
        metrics_to_log = config.metrics

    #OBTAIN SAMPLES
    if hasattr(crm.dataloader_0,"test"):
        dataloader = crm.dataloader_1.test()
    else:
        dataloader = crm.dataloader_1

    test_sample = sample_from_dataloader(dataloader,sample_size=config.maximum_test_sample_size).to(crm.device)
    generative_sample,generative_path,ts = crm.pipeline(sample_size=test_sample.shape[0],return_intermediaries=True)
    size_ = min(generative_sample.size(0),test_sample.size(0))

    generative_sample = generative_sample[:size_]
    test_sample = test_sample[:size_]

    # HISTOGRAMS
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

    metric_string_name = "nist_plot"
    if metric_string_name in metrics_to_log:
        pass

    metric_string_name = "categorical_histograms"
    if metric_string_name in metrics_to_log:
        histogram0 = categorical_histogram_dataloader(crm.dataloader_0, config.number_of_spins, config.number_of_states,
                                                      maximum_test_sample_size=config.maximum_test_sample_size)
        histogram1 = categorical_histogram_dataloader(crm.dataloader_1, config.number_of_spins, config.number_of_states,
                                                      maximum_test_sample_size=config.maximum_test_sample_size)

        generative_histogram = F.one_hot(generative_sample.long(),config.number_of_states).sum(axis=0)
        generative_histogram = generative_histogram/generative_sample.size(0)
        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = crm.experiment_files.plot_path.format("categorical_histograms_{0}".format(epoch))
        plot_categorical_histogram_per_dimension(histogram0, histogram1, generative_histogram,save_path=plot_path, remove_ticks=False)

    metric_string_name = "binary_paths_histograms"
    if metric_string_name in metrics_to_log:
        assert crm.config.number_of_states == 2
        histograms_generative = generative_path.mean(axis=0)
        if key_in_dict(where_to_log,metric_string_name):
            plot_path = where_to_log[metric_string_name]
        else:
            plot_path = crm.experiment_files.plot_path.format("binary_path_histograms_{0}".format(epoch))
        rate_logits = classification_path(crm.backward_rate, test_sample, ts)
        rate_probabilities = F.softmax(rate_logits, dim=2)[:,:,1]
        histograms_per_time_step(histograms_generative,rate_probabilities,ts,save_path=plot_path)


    return all_metrics