import torch
import unittest


class TestDataLoader(unittest.TestCase):

    def test_nist(self):
        from conditional_rate_matching.configs.config_crm import Config
        from conditional_rate_matching.data.image_dataloaders import get_data

        config = Config()
        get_data(config)

if __name__=="__main__":
    unittest.main()