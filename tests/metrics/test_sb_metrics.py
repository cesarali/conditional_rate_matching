from graph_bridges.models.metrics.sb_metrics import  paths_marginal_histograms
from graph_bridges.configs.graphs.graph_config_sb import SBConfig



import os
import torch
import unittest

from pathlib import Path
from graph_bridges.models.generative_models.sb import SB
from graph_bridges.data.graph_dataloaders_config import EgoConfig
from graph_bridges.models.backward_rates.ctdd_backward_rate_config import BackRateMLPConfig
from graph_bridges.configs.config_sb import  ParametrizedSamplerConfig, SteinSpinEstimatorConfig


class TestSB(unittest.TestCase):
    """
    Test the Schrödinger Bridge Generative Model with a super basic MLP as

    backward rate model
    """
    sb_config: SBConfig
    sb: SB

    def setUp(self) -> None:
        self.sb_config = SBConfig(experiment_indentifier="sb_unittest")
        self.sb_config.data = EgoConfig(as_image=False, batch_size=5, full_adjacency=False)
        self.sb_config.flip_estimator = SteinSpinEstimatorConfig(stein_sample_size=5)
        self.sb_config.sampler = ParametrizedSamplerConfig(num_steps=5)

        self.device = torch.device("cpu")

        self.sb = SB()
        self.sb.create_new_from_config(self.sb_config, self.device)
        databatch = next(self.sb.data_dataloader.train().__iter__())
        self.x_ajd = databatch[0]
        self.x_features = databatch[1]

    def test_marginal_histograms(self):
        all_histograms = paths_marginal_histograms(sb=self.sb,
                                                   sinkhorn_iteration=0,
                                                   device=self.device,
                                                   current_model=self.sb.training_model,
                                                   past_to_train_model=None)

        marginal_0, marginal_1, backward_histogram, forward_histogram, forward_time, state_legends = all_histograms

        expected_size = torch.Size([self.sb_config.sampler.num_steps+1,
                                    self.sb.data_dataloader.number_of_spins])
        expected_time_size = torch.Size([self.sb_config.sampler.num_steps+1])

        self.assertTrue(expected_size == backward_histogram.shape)
        self.assertTrue(expected_size == forward_histogram.shape)
        self.assertTrue(expected_time_size == forward_time.shape)

    """
    @unittest.skip
    def test_graph_metrics_and_paths_histograms(self):
        from conditional_rate_matching import results_path
        plots_paths = os.path.join(results_path,"histogram_test.png")
        plots_paths = Path(plots_paths)

        expected_path_histogram_size = torch.Size([self.sb_config.sampler.num_steps+1,
                                                   self.sb_config.data.number_of_spins])
        if plots_paths.exists():
            os.remove(plots_paths)
        stast_ = marginal_paths_histograms_plots(sb=self.sb,
                                                 sinkhorn_iteration=0,
                                                 device=self.device,
                                                 current_model=self.sb.training_model,
                                                 past_to_train_model=None,
                                                 plot_path=plots_paths)
        marginal_0, marginal_1, backward_histogram, forward_histogram, forward_time = stast_
        self.assertTrue(marginal_0.min() >= 0.)
        self.assertTrue(marginal_1.min() >= 0.)
        self.assertTrue((backward_histogram.min() >= 0.).all())
        self.assertTrue((forward_histogram.min() >= 0.).all())
        self.assertTrue(backward_histogram.shape == expected_path_histogram_size)
        self.assertTrue(forward_histogram.shape == expected_path_histogram_size)
        # check plot was performed
        self.assertTrue(plots_paths.exists())
    """

if __name__ == '__main__':
    unittest.main()
