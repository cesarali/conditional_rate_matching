import torch
import unittest
from conditional_rate_matching.models.generative_models.ctdd import CTDD


class TestMLPNist(unittest.TestCase):

    def test_mlp_mnist(self):
        from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig
        from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

        ctdd_config = CTDDConfig()
        ctdd_config.data = NISTLoaderConfig(batch_size=24)

        ctdd_config.data.batch_size = 12
        device = torch.device(ctdd_config.trainer.device)

        ctdd = CTDD(config=ctdd_config)

        databatch = next(ctdd.dataloader_0.train().__iter__())
        x_adj = databatch[0].to(device)
        batch_size = x_adj.shape[0]
        fake_time = torch.rand(batch_size).to(device)

        print(f"Data Sample Shape {x_adj.shape}")
        forward_ = ctdd.backward_rate(x_adj, fake_time)

        x_sample = ctdd.pipeline(ctdd.backward_rate, sample_size=batch_size, device=device)

        print(f"Forward Backward Rate Model {forward_.shape}")
        print(f"Pipeline sample shape {x_adj.shape}")
        print(f"X Sample {x_sample.shape}")


if __name__=="__main__":
    unittest.main()