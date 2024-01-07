import torch
import torch.nn.functional as F

from typing import Union

from conditional_rate_matching.utils.graph_utils import mask_adjs, pow_tensor,mask_x,node_feature_to_matrix
from conditional_rate_matching.models.temporal_networks.temporal_gnn.temporal_attention import TemporalAttentionLayer
from conditional_rate_matching.models.temporal_networks.temporal_gnn.temporal_layers import TemporalDenseGCNConv, MLP,TemporalMLP

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.config_ctdd import CTDDConfig
from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.utils.graph_utils import init_features
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig

class BaselineNetworkLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):

        super(BaselineNetworkLayer, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.convs.append(TemporalDenseGCNConv(conv_input_dim, conv_output_dim))
        self.hidden_dim = max(input_dim, output_dim)
        self.mlp_in_dim = input_dim + 2 * conv_output_dim
        self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim,
                       use_bn=False, activate_func=F.elu)
        self.multi_channel = MLP(2, input_dim * conv_output_dim, self.hidden_dim, conv_output_dim,
                                 use_bn=False, activate_func=F.elu)

    def forward(self, x, adj, flags):

        x_list = []
        for _ in range(len(self.convs)):
            _x = self.convs[_](x, adj[:, _, :, :])
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        x_matrix = node_feature_to_matrix(x_out)
        mlp_in = torch.cat([x_matrix, adj.permute(0, 2, 3, 1)], dim=-1)
        shape = mlp_in.shape
        mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
        _adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0, 3, 1, 2)
        _adj = _adj + _adj.transpose(-1, -2)
        adj_out = mask_adjs(_adj, flags)

        return x_out, adj_out

class BaselineNetwork(torch.nn.Module):

    def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears,
                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

        super(BaselineNetwork, self).__init__()

        self.nfeat = max_feat_num
        self.max_node_num = max_node_num
        self.nhid = nhid
        self.num_layers = num_layers
        self.num_linears = num_linears
        self.c_init = c_init
        self.c_hid = c_hid
        self.c_final = c_final

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(
                    BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))

            elif _ == self.num_layers - 1:
                self.layers.append(
                    BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))

            else:
                self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2 * self.fdim, output_dim=1,
                         use_bn=False, activate_func=F.elu)
        self.mask_diagonal_off = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask_diagonal_off.unsqueeze_(0)

    def forward(self, x, adj, flags=None):

        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, flags)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs).view(*out_shape)

        self.mask_diagonal_off = self.mask_diagonal_off.to(score.device)
        score = score * self.mask_diagonal_off

        score = mask_adjs(score, flags)

        return score

class TemporalScoreNetworkA(BaselineNetwork):

    def __init__(self, config: Union[CTDDConfig,CRMConfig,OopsConfig]):
        temporal_config = config.temporal_network

        if isinstance(config.data0,GraphDataloaderConfig):
            graph_data_config = config.data0
        elif isinstance(config.data1,GraphDataloaderConfig):
            graph_data_config = config.data1

        self.max_feat_num = graph_data_config.max_node_num
        self.max_node_num = graph_data_config.max_node_num
        self.init = graph_data_config.init

        self.nhid = temporal_config.nhid
        self.num_layers = temporal_config.num_layers
        self.num_linears = temporal_config.num_linears
        self.c_init = temporal_config.c_init
        self.c_hid = temporal_config.c_hid
        self.c_final = temporal_config.c_final
        self.adim = temporal_config.adim
        self.num_heads = temporal_config.num_heads
        self.conv = temporal_config.conv

        self.time_embed_dim = temporal_config.time_embed_dim

        super(TemporalScoreNetworkA, self).__init__(self.max_feat_num,
                                                    self.max_node_num,
                                                    self.nhid,
                                                    self.num_layers,
                                                    self.num_linears,
                                                    self.c_init,
                                                    self.c_hid,
                                                    self.c_final,
                                                    self.adim,
                                                    num_heads=self.num_heads,
                                                    conv=self.conv)

        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            if _ == 0:
                self.layers.append(TemporalAttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init,
                                                          self.c_hid, self.num_heads, self.conv,time_embed_dim=self.time_embed_dim))
            elif _ == self.num_layers - 1:
                self.layers.append(TemporalAttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                          self.c_final, self.num_heads, self.conv,time_embed_dim=self.time_embed_dim))
            else:
                self.layers.append(TemporalAttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid,
                                                          self.c_hid, self.num_heads, self.conv,time_embed_dim=self.time_embed_dim))

        self.fdim = self.c_hid * (self.num_layers - 1) + self.c_final + self.c_init
        self.final = TemporalMLP(num_layers=3,
                                 input_dim=self.fdim,
                                 hidden_dim=2 * self.fdim,
                                 output_dim=1,
                                 use_bn=False,
                                 temp_dim=self.time_embed_dim,
                                 activate_func=F.elu)

        self.mask_diagonal_off = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
        self.mask_diagonal_off.unsqueeze_(0)

        self.expected_output_shape = [self.max_node_num,self.max_node_num]

    def forward(self,adj,time):
        """
        Different from GDSS here we remove the flags (not used in the attention layers anyway)
        """
        x = init_features(self.init, adj, self.max_feat_num)
        adjc = pow_tensor(adj, self.c_init)

        adj_list = [adjc]
        for _ in range(self.num_layers):
            x, adjc = self.layers[_](x, adjc, time)
            adj_list.append(adjc)

        adjs = torch.cat(adj_list, dim=1).permute(0, 2, 3, 1)
        out_shape = adjs.shape[:-1]  # B x N x N
        score = self.final(adjs,time).view(*out_shape)

        self.mask_diagonal_off = self.mask_diagonal_off.to(score.device)
        score = score * self.mask_diagonal_off
        return score
