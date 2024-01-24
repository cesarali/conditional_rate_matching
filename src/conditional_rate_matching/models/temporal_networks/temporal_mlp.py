import torch
import numpy as np
from torch import nn as nn

import torch.nn.functional as F
from conditional_rate_matching.utils.activations import get_activation_function
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig as CRMConfig
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding

class TemporalDeepMLP(nn.Module):

    def __init__(self,
                 config,
                 device):

        super().__init__()
        self.dimensions = config.data1.dimensions
        self.vocab_size = config.data1.vocab_size
        self.define_deep_models(config)
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [self.dimensions, self.vocab_size]

    def define_deep_models(self, config):
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.num_layers = config.temporal_network.num_layers
        self.act_fn = get_activation_function(config.temporal_network.activation)
        self.dropout_rate = config.temporal_network.dropout  # Assuming dropout rate is specified in the config

        layers = [nn.Linear(self.dimensions + self.time_embed_dim, self.hidden_layer),
                  nn.BatchNorm1d(self.hidden_layer),
                  self.act_fn]

        if self.dropout_rate: layers.append(nn.Dropout(self.dropout_rate))  # Adding dropout if specified

        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_layer, self.hidden_layer),
                           nn.BatchNorm1d(self.hidden_layer),
                           self.act_fn])
            if self.dropout_rate: layers.extend([nn.Dropout(self.dropout_rate)])  # Adding dropout

        layers.append(nn.Linear(self.hidden_layer, self.dimensions * self.vocab_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)
        x = torch.concat([x, time_embeddings], dim=1)
        rate_logits = self.model(x)
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)

        return rate_logits

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)


class TemporalLeNet5(nn.Module):

    def __init__(self,
                 config,
                 device):
        super().__init__()
        self.dimensions = config.data0.dimensions
        self.vocab_size = config.data0.vocab_size
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.define_deep_models()
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [28, 28, self.vocab_size]

    def define_deep_models(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn2d1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2d2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4 + self.time_embed_dim, self.hidden_layer)
        self.bn1 = nn.BatchNorm1d(self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.hidden_layer)
        self.bn2 = nn.BatchNorm1d(self.hidden_layer)
        self.fc3 = nn.Linear(self.hidden_layer + self.time_embed_dim, 28 * 28 * 2)

    def forward(self, x, times):
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)

        x = F.max_pool2d(F.relu(self.bn2d1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2d2(self.conv2(x))), (2, 2))
        x = x.view(-1, np.prod(x.size()[1:]))

        x = torch.concat([x, time_embeddings], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))

        x = torch.concat([x, time_embeddings], dim=1)
        x = F.relu(self.bn2(self.fc2(x)))

        x = torch.concat([x, time_embeddings], dim=1)
        x = self.fc3(x)

        return x.view(-1, 28, 28, 2)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)




class TemporalMLP(nn.Module):
    """
    """
    def __init__(self, config:CRMConfig, device):
        super().__init__()
        if hasattr(config,'data1'):
            config_data = config.data1
        else:
            config_data = config.data0

        self.dimensions = config_data.dimensions
        self.vocab_size = config_data.vocab_size

        self.define_deep_models(config)
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [self.dimensions,self.vocab_size]

    def define_deep_models(self,config):
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.f1 = nn.Linear(self.dimensions, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimensions * self.vocab_size)

    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)

        return rate_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)
