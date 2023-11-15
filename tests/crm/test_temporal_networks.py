import torch
import unittest

from conditional_rate_matching.models.generative_models.crm import CRM
from experiments.testing_MNIST import experiment_MNIST_Convnet, experiment_MNIST

class TestMNISTConvNet(unittest.TestCase):

    def test_Convnet(self):
        config = experiment_MNIST_Convnet()
        crm = CRM(config=config)

        databatch = next(crm.dataloader_1.train().__iter__())
        x = databatch[0].to(crm.device)
        t = torch.rand((x.size(0),)).to(crm.device)

        batch_size = x.size(0)
        change_logits = crm.forward_rate.temporal_network(x,t)
        change_logits = change_logits.reshape(batch_size,-1)
        change_logits = crm.forward_rate.temporal_to_rate(change_logits)
        change_logits = change_logits.reshape(batch_size,crm.forward_rate.dimensions,crm.forward_rate.vocab_size)
        print(change_logits.shape)

        rate = crm.forward_rate(x,t)
        print(rate.shape)

class TestMNIST(unittest.TestCase):

    def test_mlp(self):
        config = experiment_MNIST()
        crm = CRM(config=config)

        databatch = next(crm.dataloader_1.train().__iter__())
        x = databatch[0].to(crm.device)
        t = torch.rand((x.size(0),)).to(crm.device)

        rate = crm.forward_rate(x,t)
        print(rate.shape)


if __name__=="__main__":
    unittest.main()