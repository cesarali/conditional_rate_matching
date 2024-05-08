import os
import torch
import unittest
from conditional_rate_matching.configs.config_files import get_experiment_dir
from conditional_rate_matching.models.generative_models.ctdd import CTDD
from conditional_rate_matching.utils.devices import check_model_devices
class TestLoading(unittest.TestCase):

    def test_loading(self):
        experiment_dir = get_experiment_dir(experiment_name="harz_experiment",
                                            experiment_type="ctdd",
                                            experiment_indentifier="test")
        ctdd =  CTDD(experiment_dir=experiment_dir)
        databatch = next(ctdd.dataloader_0.train().__iter__())

        device = check_model_devices(ctdd.backward_rate)
        x = databatch[0].to(device)
        batchsize = x.size(0)
        times_ = torch.rand((batchsize,)).to(device)
        forward_rates,qt0_denom,qt0_numer = ctdd.process.forward_rates_and_probabilities(x,times_)
        print(forward_rates.shape)
        print(qt0_denom.shape)



if __name__=="__main__":
    unittest.main()