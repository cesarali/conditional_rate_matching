import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from conditional_rate_matching.configs.config_crm import CRMConfig as CRMConfig
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding
from conditional_rate_matching.utils.activations import get_activation_function

class TemporalDeepEBM(nn.Module):

    def __init__(self,
                 config,
                 device):

        super().__init__()
        self.device=device
        self.dimensions = config.data0.dimensions
        self.vocab_size = config.data0.vocab_size
        self.define_deep_models(config)
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [self.dimensions, self.vocab_size]

    def define_deep_models(self, config):
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.num_layers = config.temporal_network.num_layers
        self.act_fn = get_activation_function(config.temporal_network.activation)
        self.dropout_rate = config.temporal_network.dropout  
        layers = []
        batchnorm_layers = []

        # First layer
        layers.append(nn.Linear(self.dimensions + self.time_embed_dim, self.hidden_layer))
        batchnorm_layers.append(nn.BatchNorm1d(self.hidden_layer))
        if self.dropout_rate: batchnorm_layers.append(nn.Dropout(self.dropout_rate))

        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_layer + self.time_embed_dim, self.hidden_layer))
            batchnorm_layers.append(nn.BatchNorm1d(self.hidden_layer))
            if self.dropout_rate: batchnorm_layers.append(nn.Dropout(self.dropout_rate))

        self.layers = nn.ModuleList(layers)
        self.batchnorm_layers = nn.ModuleList(batchnorm_layers)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimensions * self.vocab_size)


    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = torch.concat([x, time_embeddings], dim=1)
                x = layer(x)
                x = self.batchnorm_layers[i](x)
                x = self.batchnorm_layers[i+1](x)
                if self.act_fn: x = self.act_fn(x)

        x = torch.concat([x, time_embeddings], dim=1)
        rate_logits = self.output_layer(x)
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)
        return rate_logits

    def init_weights(self):
        for layer in self.layers + [self.output_layer]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
