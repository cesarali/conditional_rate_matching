import unittest


import torch
from conditional_rate_matching.models.pipelines.pipeline_crm import CRMPipeline
from conditional_rate_matching.models.generative_models.crm import ClassificationForwardRate

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm

class TestCRMPipeline(unittest.TestCase):

    def test_pipeline_classifier(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        config = CRMConfig()
        dataloader_0, dataloader_1 = get_dataloaders_crm(config)
        config.loss = "classifier"
        model = ClassificationForwardRate(config, device).to(device)

        pipeline = CRMPipeline(config,model,dataloader_0,dataloader_1)
        x_f = pipeline(sample_size=132)
        print(x_f.shape)


if __name__=="__main__":
    unittest.main()