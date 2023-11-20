import unittest
import torch.cuda

from conditional_rate_matching.configs.config_ctdd import CTDDConfig
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_ctdd
from conditional_rate_matching.models.temporal_networks.rates.ctdd_rates import BackRateMLP
from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import GaussianTargetRate

class TestGaussianReference(unittest.TestCase):

    def setUp(self) -> None:
        config = CTDDConfig()

        self.dataloader_0,self.dataloader_1 = get_dataloaders_ctdd(config)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.x_adj_data = next(self.dataloader_0.train().__iter__())[0].to(self.device)
        self.x_adj_target = next(self.dataloader_1.train().__iter__())[0].to(self.device)

        self.batch_size = self.x_adj_data.shape[0]
        self.times = torch.rand(self.batch_size,device=self.device)

        self.reference_process = GaussianTargetRate(config, self.device)

    def test_rates_and_probabilities(self):
        forward_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)
        rate_ = self.reference_process.rate(self.times)
        transition_ = self.reference_process.transition(self.times)
        forward_rates_and_probabilities_ = self.reference_process.forward_rates_and_probabilities(self.x_adj_data, self.times, self.device)
        forward_rates, qt0_denom, qt0_numer = forward_rates_and_probabilities_
        forward_rates_and_probabilities_ = self.reference_process.forward_rates_and_probabilities(self.x_adj_target, self.times, self.device)
        forward_rates, qt0_denom, qt0_numer = forward_rates_and_probabilities_
        forward_rates_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)
        forward_rates_ = self.reference_process.forward_rates(self.x_adj_target, self.times, self.device)
        print(
            "rate_ {0}, transition_ {1}, forward_rates {2}, qt0_denom {3}, qt0_numer {4}, forward_rates_ {5}, ".format(
                rate_.shape, transition_.shape, forward_rates.shape, qt0_denom.shape, qt0_numer.shape,
                forward_rates_.shape))

    def test_rates_and_probabilities_device(self):
        forward_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)
        rate_ = self.reference_process.rate(self.times)
        transition_ = self.reference_process.transition(self.times)
        forward_rates_and_probabilities_ = self.reference_process.forward_rates_and_probabilities(self.x_adj_data, self.times, self.device)
        forward_rates, qt0_denom, qt0_numer = forward_rates_and_probabilities_
        forward_rates_ = self.reference_process.forward_rates(self.x_adj_data, self.times, self.device)

        self.assertTrue(self.device == stein_binary_forward.device)
        self.assertTrue(self.device == forward_.device)
        self.assertTrue(self.device == rate_.device)
        self.assertTrue(self.device == transition_.device)
        self.assertTrue(self.device == forward_rates.device)
        self.assertTrue(self.device == qt0_denom.device)
        self.assertTrue(self.device == qt0_numer.device)
        self.assertTrue(self.device == forward_rates_.device)


if __name__=="__main__":
    print("Hello!")