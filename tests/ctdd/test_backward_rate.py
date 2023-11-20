import unittest
import torch.cuda

from conditional_rate_matching.configs.config_ctdd import CTDDConfig
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_ctdd
from conditional_rate_matching.models.temporal_networks.rates.ctdd_rates import BackRateMLP

class TestBackwardRate(unittest.TestCase):

    def test_rate(self):
        config = CTDDConfig()
        print(config)

        dataloader_0,dataloader_1 = get_dataloaders_ctdd(config)
        databatch1 = next(dataloader_1.train().__iter__())
        adj_x = databatch1[0]

        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        backward_rate = BackRateMLP(config,device)

        #batch_size = adj_x.size(0)
        #times_ = torch.rand((batch_size))
        #print(times_.shape)




if __name__=="__main__":
    print("Hello!")