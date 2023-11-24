
import torch
from torch import nn
from conditional_rate_matching.models.temporal_networks.embedding_utils import transformer_timestep_embedding



class TemporalDeepMLP(nn.Module):

    def __init__(self, 
                 config, 
                 device):
        
        super().__init__()
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
        self.activation_fn = get_activation_function(config.temporal_network.activation)

        layers = [nn.Linear(self.dimensions + self.time_embed_dim, self.hidden_layer)]
        if self.activation_fn: layers.append(self.activation_fn)
        
        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_layer, self.hidden_layer)])
            if self.activation_fn: layers.extend([self.activation_fn])

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







class TemporalDeepSets(nn.Module):
    pass