import math
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from conditional_rate_matching.utils.graph_utils import mask_adjs, mask_x
from conditional_rate_matching.models.temporal_networks.temporal_gnn.temporal_layers import TemporalDenseGCNConv, MLP

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

        V = self.gnn_v(x, adj,time)
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


# -------- Layer of ScoreNetworkA --------
class TemporalAttentionLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim,
                 num_heads=4, conv='GCN',time_embed_dim=19):

        super(TemporalAttentionLayer, self).__init__()

        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn_dim = attn_dim
            self.attn.append(TemporalAttention(conv_input_dim, self.attn_dim, conv_output_dim,
                                               num_heads=num_heads, conv=conv,time_embed_dim=time_embed_dim))

        self.hidden_dim = 2 * max(input_dim, output_dim)
        self.mlp = MLP(num_linears, 2 * input_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim * conv_output_dim, self.hidden_dim, conv_output_dim,
                                 use_bn=False, activate_func=F.elu)

    def forward(self, x, adj, time):
        """

        :param x:  B x N x F_i
        :param adj: B x C_i x N x N
        :return: x_out: B x N x F_o, adj_out: B x C_o x N x N
        """
        mask_list = []
        x_list = []
        for _ in range(len(self.attn)):
            _x, mask = self.attn[_](x, adj[:, _, :, :], time)
            mask_list.append(mask.unsqueeze(-1))
            x_list.append(_x)
        x_out = self.multi_channel(torch.cat(x_list, dim=-1))
        x_out = torch.tanh(x_out)

        mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        adj_out = _adj + _adj.transpose(-1, -2)

        return x_out, adj_out
