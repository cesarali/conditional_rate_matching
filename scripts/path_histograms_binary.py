import torch

# configs
from conditional_rate_matching.configs.config_crm import NistConfig

# data
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader

# models
from conditional_rate_matching.models.generative_models.crm import (
    ConditionalBackwardRate,
    ClassificationForwardRate,
    uniform_pair_x0_x1
)

# pipelines
from conditional_rate_matching.models.pipelines.mc_samplers import TauLeaping

# metrics
from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader
from conditional_rate_matching.models.metrics.crm_path_metrics import telegram_bridge_sample_paths

# plots
from conditional_rate_matching.utils.plots.histograms_plots import plot_marginals_binary_histograms

if __name__ == "__main__":
    config = NistConfig(batch_size=128)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # conditional model
    conditional_model = ConditionalBackwardRate(config, device)
    classification_model = ClassificationForwardRate(config, device).to(device)

    # data
    dataloader_0, dataloader_1 = get_dataloaders(config)
    batch_0, batch_1 = next(zip(dataloader_0, dataloader_1).__iter__())
    x_1, x_0 = uniform_pair_x0_x1(batch_1, batch_0, device)
    x_f, x_hist, x0_hist, ts = TauLeaping(config, classification_model, x_0, forward=True)

    # histograms
    histogram0 = binary_histogram_dataloader(dataloader_0, dimensions=config.dimension,
                                             train=True, maximum_test_sample_size=config.maximum_test_sample_size)
    histogram1 = binary_histogram_dataloader(dataloader_1, dimensions=config.dimension,
                                             train=True, maximum_test_sample_size=config.maximum_test_sample_size)
    marginal_histograms = (histogram0, torch.zeros_like(histogram0), histogram1, torch.zeros_like(histogram1))
    plot_marginals_binary_histograms(marginal_histograms)

    X_0 = sample_from_dataloader(dataloader_0,sample_size=250).to(device)
    X_1 = sample_from_dataloader(dataloader_1,sample_size=250).to(device)
    time_steps = torch.linspace(0.,1.,10).to(device)

    telegram_histograms_path,time_grid = telegram_bridge_sample_paths(config,X_0,X_1,time_steps,histogram=True)

    #histograms_per_time_step(telegram_histograms_path,None,time_grid)

    


