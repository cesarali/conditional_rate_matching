import torch
from conditional_rate_matching.models.generative_models.crm import CRM
def log_metrics(crm: CRM,epoch, device, metrics_to_log=None, where_to_log=None, writer=None):
    """
    After the training procedure is done, the model is updated

    :return:
    """
    all_metrics = {}

    config = oops.config
    if metrics_to_log is None:
        metrics_to_log = config.trainer.metrics

    #OBTAIN SAMPLES
    test_sample = torch.vstack([databatch[0] for databatch in dataloader.test()])
    generative_sample = oops.pipeline(sample_size=test_sample.shape[0])

    # HISTOGRAMS
    metric_string_name = "mse_histograms"
    if metric_string_name in metrics_to_log:
        mse_marginal_histograms = marginal_histograms(generative_sample,test_sample)
        mse_metrics = {"mse_marginal_histograms": mse_marginal_histograms.tolist()}
        all_metrics = store_metrics(config, all_metrics, new_metrics=mse_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)

    metric_string_name = "kdmm"
    if metric_string_name in metrics_to_log:
        mse_0 = kmmd(generative_sample,test_sample)
        mse_metrics = {"mse_histograms_0": mse_0.item()}
        all_metrics = store_metrics(config, all_metrics, new_metrics=mse_metrics, metric_string_name=metric_string_name, epoch=epoch, where_to_log=where_to_log)
