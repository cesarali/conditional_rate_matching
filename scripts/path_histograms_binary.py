import torch

# configs
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

# data
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader

# models
from conditional_rate_matching.models.generative_models.crm import (
    CRM,
)

# metrics
from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader
from conditional_rate_matching.models.metrics.crm_path_metrics import telegram_bridge_sample_paths

# plots
from conditional_rate_matching.utils.plots.histograms_plots import plot_marginals_binary_histograms

from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping

if __name__ == "__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    from conditional_rate_matching.utils.plots.images_plots import plot_sample
    from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist
    from conditional_rate_matching.utils.plots.paths_plots import histograms_per_time_step
    config : CRMConfig
    config = experiment_nist(dataset_name="mnist",temporal_network_name="mlp")
    crm = CRM(config)

    databatch_0 = next(crm.dataloader_0.train().__iter__())
    x_0 = databatch_0[0]

    databatch_1 = next(crm.dataloader_1.train().__iter__())
    x_1 = databatch_1[0]

    # rate_model = lambda x, t: constant_rate(config, x, t)
    #rate_model = lambda x, t: crm.forward_rate.conditional_transition_rate(x, x_1, t)
    #x_f, x_hist, x0_hist, ts = TauLeaping(config, rate_model, x_0, forward=True)

#    print(x_hist.shape)
#    plot_sample(x_0)
#    plot_sample(x_f)

    # histograms
    histogram0 = binary_histogram_dataloader(crm.dataloader_0, dimensions=config.data1.dimensions,
                                             train=True, maximum_test_sample_size=config.trainer.max_test_size)
    histogram1 = binary_histogram_dataloader(crm.dataloader_1, dimensions=config.data1.dimensions,
                                             train=True, maximum_test_sample_size=config.trainer.max_test_size)
    marginal_histograms = (histogram0, torch.zeros_like(histogram0), histogram1, torch.zeros_like(histogram1))
    plot_marginals_binary_histograms(marginal_histograms)

    X_0 = sample_from_dataloader(crm.dataloader_0.train(),sample_size=250).to(device)
    X_1 = sample_from_dataloader(crm.dataloader_1.train(),sample_size=250).to(device)
    time_steps = torch.linspace(0.,1.,10).to(device)

    telegram_histograms_path,time_grid = telegram_bridge_sample_paths(crm,X_0,X_1,time_steps,histogram=True)

    histograms_per_time_step(telegram_histograms_path,None,time_grid)
