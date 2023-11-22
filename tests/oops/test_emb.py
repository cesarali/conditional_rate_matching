import unittest

import torch

from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.models.temporal_networks.mlp_utils import get_net
from conditional_rate_matching.models.temporal_networks.ebm import EBM

from conditional_rate_matching.data.dataloaders_utils import get_dataloader_oops

class TestEMB(unittest.TestCase):

    def test_emb(self):
        configs = OopsConfig()
        device = torch.device(configs.trainer.device) if torch.cuda.is_available() else torch.device("cpu")
        dataloader = get_dataloader_oops(configs)
        databatch = next(dataloader.train().__iter__())
        x = databatch[0].to(device)
        net = get_net(configs,device)
        h = net(x)
        print(h.shape)

        ebm = EBM(configs,device=device).to(device)
        h = ebm(x)
        print(h.shape)





if __name__ == "__main__":
    unittest.main()