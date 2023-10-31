import torch
from torch import nn
from conditional_rate_matching.models.temporal_networks.embedding_utils import transformer_timestep_embedding
from torch.nn.functional import softplus,softmax

class TemporalMLP(nn.Module):

    def __init__(self, dimensions, number_of_states, time_embed_dim, hidden_dim, device):
        super().__init__()

        self.time_embed_dim = time_embed_dim
        self.hidden_layer = hidden_dim
        self.num_states = number_of_states
        self.dimension = dimensions
        self.expected_output_shape = [self.dimension, self.num_states]

        self.define_deep_models()
        self.init_weights()

        self.device = device
        self.to(self.device)

    def define_deep_models(self):
        # layers
        self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimension * self.num_states)

    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size, self.dimension, self.num_states)

        return rate_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

def beta_integral(gamma, t1, t0):
    """
    Dummy integral for constant rate
    """
    interval = t1 - t0
    integral = gamma * interval
    return integral


class ClassificationBackwardRate(nn.Module):

    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.S = config.number_of_states
        self.D = config.number_of_spins
        self.time_embed_dim = config.time_embed_dim
        self.hidden_layer = config.hidden_dim
        self.dimension = self.D
        self.num_states = self.S

        self.expected_data_shape = [config.number_of_spins]
        self.define_deep_models()
        self.init_weights()
        self.to(device)

    def define_deep_models(self):
        self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimension * self.num_states)

        # self.f1 = nn.Linear(self.dimension, self.hidden_layer)
        # self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimension * self.num_states)

    def to_go(self, x, t):
        batch_size = x.size(0)
        x_to_go = torch.arange(0, self.S)
        x_to_go = x_to_go[None, None, :].repeat((batch_size, self.D, 1)).float()
        x_to_go = x_to_go.to(x.device)
        return x_to_go

    def classify(self, x, times):
        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size, self.dimension, self.num_states)

        return rate_logits

    def forward(self, x, time):
        right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

        batch_size = x.size(0)

        w_1t = beta_integral(self.config.gamma, right_time_size(1.), right_time_size(time))
        A = 1.
        B = (w_1t * self.S) / (1. - w_1t)
        C = w_1t

        x_to_go = self.to_go(x, time)
        x_to_go = x_to_go.view((batch_size * self.S, self.D))

        rate_logits = self.classify(x, time)
        rate_probabilities = softmax(rate_logits)
        rates = A + B[:,None,None]*rate_probabilities +  C[:,None,None]*rate_probabilities

        return rates

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

class ConditionalBackwardRate(nn.Module):
    """

    """
    def __init__(self, config, device):
        super().__init__()
        self.expected_data_shape = [config.number_of_spins]
        self.temporal_network = TemporalMLP(dimensions=config.number_of_spins,
                                            number_of_states=config.number_of_states,
                                            time_embed_dim=config.time_embed_dim,
                                            hidden_dim=config.hidden_dim,
                                            device=device).to(device)
        # self.logits_to_rates = nn.Linear(self.temporal_network_output_size,)

    def forward(self, x, time):
        batch_size = x.size(0)

        # ================================
        expected_data_shape_ = torch.Size([batch_size] + self.expected_data_shape)
        temporal_network_logits = self.temporal_network(x, time)
        rates_ = softplus(temporal_network_logits)
        return rates_