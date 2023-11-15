import torch
from torch import nn
from torch.nn.functional import softplus,softmax
from conditional_rate_matching.configs.config_crm import Config

from conditional_rate_matching.models.temporal_networks.embedding_utils import transformer_timestep_embedding
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network
from functools import reduce


def flip_rates(conditional_model,x_0,time):
    conditional_rate = conditional_model(x_0, time)
    not_x_0 = (~x_0.bool()).long()
    flip_rate = torch.gather(conditional_rate, 2, not_x_0.unsqueeze(2)).squeeze()
    return flip_rate

def beta_integral(gamma, t1, t0):
    """
    Dummy integral for constant rate
    """
    interval = t1 - t0
    integral = gamma * interval
    return integral

class ClassificationForwardRate(nn.Module):
    """

    """
    def __init__(self, config:Config, device):
        super().__init__()

        self.config = config
        self.vocab_size = config.data1.vocab_size
        self.dimensions = config.data1.dimensions

        self.expected_data_shape = config.data1.temporal_net_expected_shape

        self.define_deep_models(config,device)
        self.to(device)

    def define_deep_models(self,config,device):
        self.temporal_network = load_temporal_network(config,device=device)
        self.expected_temporal_output_shape = self.temporal_network.expected_output_shape
        if self.expected_temporal_output_shape != [self.dimensions,self.vocab_size]:
            temporal_output_total = reduce(lambda x, y: x * y, self.expected_temporal_output_shape)
            self.temporal_to_rate = nn.Linear(temporal_output_total,self.dimensions*self.vocab_size)

    def classify(self,x,times):
        """
        this function takes the shape [batch_size,dimension,vocab_size] and make all the trsformations
        to handle the temporal network

        :param x: [batch_size,dimension,vocab_size]
        :param times:
        :return:
        """
        batch_size = x.size(0)
        expected_shape_for_temporal = torch.Size([batch_size]+self.expected_data_shape)
        current_shape = x.shape
        if current_shape != expected_shape_for_temporal:
            x = x.reshape(expected_shape_for_temporal)
        change_logits = self.temporal_network(x,times)

        if self.temporal_network.expected_output_shape != [self.dimensions,self.vocab_size]:
            change_logits = change_logits.reshape(batch_size, -1)
            change_logits = self.temporal_to_rate(change_logits)
            change_logits = change_logits.reshape(batch_size,self.dimensions,self.vocab_size)
        return change_logits

    def forward(self, x, time):
        """
        RATE

        :param x: [batch_size,dimensions]
        :param time:
        :return:[batch_size,dimensions,vocabulary_size]
        """
        batch_size = x.size(0)
        if len(x.shape) != 2:
            x = x.reshape(batch_size,-1)
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

        beta_integral_ = beta_integral(self.config.process.gamma, right_time_size(1.), right_time_size(time))
        w_1t = torch.exp(-self.vocab_size * beta_integral_)
        A = 1.
        B = (w_1t * self.vocab_size) / (1. - w_1t)
        C = w_1t

        change_logits = self.classify(x, time)

        change_classifier = softmax(change_logits, dim=2)
        #x = x.reshape(batch_size,self.dimensions)
        where_iam_classifier = torch.gather(change_classifier, 2, x.long().unsqueeze(2))

        rates = A + B[:,None,None]*change_classifier + C[:,None,None]*where_iam_classifier

        return rates
