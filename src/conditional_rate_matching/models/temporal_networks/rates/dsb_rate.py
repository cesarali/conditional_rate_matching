import torch
from torch import nn
from torch.nn.functional import softplus
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network

from functools import reduce


def flip_rates(conditional_model,x_0,time):
    conditional_rate = conditional_model(x_0, time)
    not_x_0 = (~x_0.bool()).long()
    flip_rate = torch.gather(conditional_rate, 2, not_x_0.unsqueeze(2)).squeeze()
    return flip_rate

class SchrodingerBridgeRate(nn.Module):
    """
    """
    def __init__(self, config:CRMConfig, device):
        super().__init__()

        self.config = config
        self.vocab_size = config.data0.vocab_size
        self.dimensions = config.data0.dimensions
        self.expected_data_shape = config.data0.temporal_net_expected_shape

        self.define_deep_models(config,device)
        self.to(device)

    def define_deep_models(self,config,device):
        self.temporal_network = load_temporal_network(config,device=device)
        self.expected_temporal_output_shape = self.temporal_network.expected_output_shape
        if self.expected_temporal_output_shape != [self.dimensions]:
            temporal_output_total = reduce(lambda x, y: x * y, self.expected_temporal_output_shape)
            self.temporal_to_rate = nn.Linear(temporal_output_total,self.dimensions)


    def flip_rate(self,x,times):
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
        flip_rate_logits = self.temporal_network(x,times)

        if self.temporal_network.expected_output_shape != [self.dimensions]:
            flip_rate_logits = flip_rate_logits.reshape(batch_size, -1)
            flip_rate_logits = self.temporal_to_rate(flip_rate_logits)
            flip_rate_logits = flip_rate_logits.reshape(batch_size,self.dimensions)
        flip_rate_logits = softplus(flip_rate_logits)

        return flip_rate_logits

    def forward(self, x, time):
        """
        RATE

        :param x: [batch_size,dimensions]
        :param time:
        :return:[batch_size,dimensions,vocabulary_size]
        """
        rate = self.flip_rate(x,time)[:,:,None].repeat((1,1,2))
        return rate
