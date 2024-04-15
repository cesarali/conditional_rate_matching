from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalScoreNetworkAConfig
from conditional_rate_matching.models.temporal_networks.temporal_gnn.temporal_layers import TemporalMLP,MLP,TemporalLinear
import torch.nn.functional as F
import torch
import sys
import os

if __name__=="__main__":

    batch_size = 2
    input_dim = 5
    hidden_dim = 3
    output_dim = 5
    temp_dim = 12
    use_bn = False

    x = torch.rand((batch_size,input_dim))
    time = torch.rand((batch_size,))

    mlp = MLP(2, input_dim, hidden_dim, output_dim, use_bn=use_bn, activate_func=F.elu)
    temporal_mlp = TemporalMLP(2, input_dim, hidden_dim, output_dim, use_bn=use_bn, activate_func=F.elu,temp_dim=temp_dim)

    print("Non Temporal MLP")
    output = mlp(x)
    print(output.shape)
    print("Temporal MLP")
    temp_output = temporal_mlp(x,time)
    print(temp_output.shape)
