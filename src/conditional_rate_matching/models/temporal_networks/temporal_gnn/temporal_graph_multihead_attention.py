import math
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from conditional_rate_matching.utils.graph_utils import mask_adjs, mask_x
from conditional_rate_matching.models.temporal_networks.temporal_gnn.temporal_layers import TemporalDenseGCNConv, MLP
from conditional_rate_matching.models.temporal_networks.temporal_gnn.temporal_layers import TemporalMLP



# -------- Graph Multi-Head Attention (GMH) --------
# -------- From Baek et al. (2021) --------
class TemporalAttention(torch.nn.Module):

    def __init__(self, in_dim, attn_dim, out_dim, num_heads=4, conv='GCN',time_embed_dim=19):
        super(TemporalAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv

        self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(in_dim, attn_dim, out_dim, conv,time_embed_dim=time_embed_dim)
        self.activation = torch.tanh
        self.softmax_dim = 2

    def forward(self, x, adj, time, attention_mask=None):

        if self.conv == 'GCN':
            Q = self.gnn_q(x, adj,time)
            K = self.gnn_k(x, adj,time)
        else:
            Q = self.gnn_q(x)
            K = self.gnn_k(x)

        V = self.gnn_v(x,adj,time)
        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim)
            A = self.activation(attention_mask + attention_score)
        else:
            A = self.activation(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.out_dim))  # (B x num_heads) x N x N

        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, *adj.shape)
        A = A.mean(dim=0)
        A = (A + A.transpose(-1, -2)) / 2

        return V, A

    def get_gnn(self, in_dim, attn_dim, out_dim, conv='GCN',time_embed_dim=19):

        if conv == 'GCN':
            gnn_q = TemporalDenseGCNConv(in_dim, attn_dim,time_embed_dim=time_embed_dim)
            gnn_k = TemporalDenseGCNConv(in_dim, attn_dim,time_embed_dim=time_embed_dim)
            gnn_v = TemporalDenseGCNConv(in_dim, out_dim,time_embed_dim=time_embed_dim)
            return gnn_q, gnn_k, gnn_v

        elif conv == 'MLP':
            num_layers = 2
            gnn_q = MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_k = MLP(num_layers, in_dim, 2 * attn_dim, attn_dim, activate_func=torch.tanh)
            gnn_v = TemporalDenseGCNConv(in_dim, out_dim,time_embed_dim=time_embed_dim)

            return gnn_q, gnn_k, gnn_v

        else:
            raise NotImplementedError(f'{conv} not implemented.')