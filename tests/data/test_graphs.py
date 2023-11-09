import os
import unittest

from conditional_rate_matching.data.graph_dataloaders_config import CommunityConfig,CommunitySmallConfig
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.utils.data import check_sizes


class TestGraphDataloader(unittest.TestCase):

    def test_dataloader(self):
        data_config = CommunitySmallConfig(batch_size=5, as_image=False, full_adjacency=True, flatten=False)
        data_loader = GraphDataloaders(data_config)
        databatch = next(data_loader.train().__iter__())
        print(f"databatch_shape {databatch[0].shape} expected shape {data_config.temporal_net_expected_shape} expected dim {data_config.dimensions}")

        train_size,test_size = check_sizes(data_loader)
        print(f"train size {train_size} test size {test_size} epectedt train {data_config.training_size} expected test {data_config.test_size}")


if __name__=="__main__":
    unittest.main()