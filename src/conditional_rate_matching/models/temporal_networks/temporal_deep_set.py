import torch
import numpy as np
from conditional_rate_matching.utils.activations import get_activation_function
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding

class TemporalDeepSets(torch.nn.Module):
    def __init__(self,
                 config,
                 device,
                 pool='meansum'
                 ):
        super().__init__()
        self.dimensions = int(np.sqrt(config.data0.dimensions))  # dxd -> d
        self.vocab_size = config.data0.vocab_size
        self.define_deep_models(config)
        self.to(device)
        self.expected_output_shape = [self.dimensions, self.vocab_size]

    def define_deep_models(self, config):
        self.dim_hidden_t = config.temporal_network.time_embed_dim
        self.dim_hidden_x = config.temporal_network.hidden_dim
        self.num_layers = config.temporal_network.num_layers
        self.act_fn = get_activation_function(config.temporal_network.activation)
        self.pool = config.temporal_network.pool

        s = 2 if self.pool == 'meansum' else 1

        phi_layers = [torch.nn.Linear(self.dimensions + self.dim_hidden_t, self.dim_hidden_x), self.act_fn]
        for _ in range(self.num_layers - 1): phi_layers.extend(
            [torch.nn.Linear(self.dim_hidden_x, self.dim_hidden_x), self.act_fn])
        phi_layers.append(torch.nn.Linear(self.dim_hidden_x, self.dim_hidden_x))
        self.phi = torch.nn.Sequential(*phi_layers)

        rho_layers = [torch.nn.Linear(s * self.dim_hidden_x, self.dim_hidden_x), self.act_fn]
        for _ in range(self.num_layers - 1): rho_layers.extend(
            [torch.nn.Linear(self.dim_hidden_x, self.dim_hidden_x), self.act_fn])
        rho_layers.append(torch.nn.Linear(self.dim_hidden_x, self.dimensions * self.vocab_size))
        self.rho = torch.nn.Sequential(*rho_layers)

    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.dim_hidden_t)
        time_embeddings = time_embeddings.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.concat([x, time_embeddings], dim=-1)

        # ...deepsets:
        h = self.phi(x)
        h_sum = h.sum(1, keepdim=False)
        h_mean = h.mean(1, keepdim=False)

        # ...aggregation:
        if self.pool == 'sum':
            h_pool = h_sum
        elif self.pool == 'mean':
            h_pool = h_mean
        elif self.pool == 'meansum':
            h_pool = torch.cat([h_mean, h_sum], dim=1)

        rate_logits = self.rho(h_pool)
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)
        return rate_logits
