import torch
import torch.nn as nn
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding

import numpy as np
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from conditional_rate_matching.utils.activations import get_activation_function

class TemporalGraphConvNet(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.dimensions = int(np.sqrt(config.data0.dimensions))  # dxd -> d
        self.vocab_size = config.data0.vocab_size
        self.define_deep_models(config)
        self.to(device)
        self.expected_output_shape = [self.dimensions, self.vocab_size]

    def define_deep_models(self, config):
        self.dim_hidden_t = config.temporal_network.time_embed_dim
        self.dim_hidden_x = config.temporal_network.hidden_dim
        self.act_fn = get_activation_function(config.temporal_network.activation)

        # ...define GNN layers
        self.conv1 = GCNConv(self.dim_hidden_t + 1, self.dim_hidden_x)
        self.conv2 = GCNConv(self.dim_hidden_x, self.dim_hidden_x)
        self.linear = nn.Linear(self.dim_hidden_x, self.dimensions * self.vocab_size)

    def forward(self, adj, times):
        B, N, D = adj.shape
        node_degree = adj.sum(dim=1).unsqueeze(-1)
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.dim_hidden_t)
        time_embeddings = time_embeddings.unsqueeze(1).repeat(1, N, 1)
        node_features = torch.concat([node_degree, time_embeddings], dim=-1)

        data_list = []
        for i in range(B):
            edge_index, _ = dense_to_sparse(adj[i])
            data_list.append(Data(x=node_features[i], edge_index=edge_index))

        batched_data = Batch.from_data_list(data_list)

        h = self.conv1(batched_data.x, batched_data.edge_index)
        if self.act_fn is not None: h = self.act_fn(h)
        h = self.conv2(h, batched_data.edge_index)
        if self.act_fn is not None: h = self.act_fn(h)
        h = torch_geometric.nn.global_mean_pool(h, batched_data.batch)
        rate_logits = self.linear(h)
        rate_logits = rate_logits.reshape(B, self.dimensions, self.vocab_size)
        return rate_logits