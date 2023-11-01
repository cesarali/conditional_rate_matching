import torch
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.metrics.metrics_utils import store_metrics
from conditional_rate_matching.models.metrics.distances import kmmd,marginal_histograms
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader
from conditional_rate_matching.utils.plots.histograms_plots import plot_categorical_histogram_per_dimension
import torch.nn.functional as F

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

    test_sample = torch.vstack([databatch[0] for databatch in dataloader])
    generative_sample = crm.pipeline(sample_size=test_sample.shape[0])

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

        plot_categorical_histogram_per_dimension(histogram0, histogram1, generative_histogram,remove_ticks=False)
        #writer.add_image('random_image', image_tensor)

    return all_metrics