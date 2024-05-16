
import torch
from torch import nn
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm
from torch.nn import init
import math

from torch_geometric.utils import dense_to_sparse,unbatch
from torch_geometric.data import Data
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

def sample_to_geometric(X,number_of_nodes=20):
    """
    obtains a representation which is suuitable for GNNs defined wiith the torch_geometric library
    """
    batch_size = X.shape[0]
    adj = X[:,:,None].reshape(batch_size,number_of_nodes,number_of_nodes)
    edge_index = dense_to_sparse(torch.Tensor(adj))[0]
    nodes_attributes = torch.eye(number_of_nodes)
    nodes_attributes = nodes_attributes.repeat((batch_size,1))
    batch = torch.arange(batch_size).repeat_interleave(number_of_nodes)
    return nodes_attributes,edge_index,batch
    
class SimpleTemporalGCN(torch.nn.Module):
    def __init__(self,config:CRMConfig):
        super(SimpleTemporalGCN, self).__init__()

        self.number_of_nodes = config.data1.max_node_num
        self.num_node_features = self.number_of_nodes 
        self.hidden_channels = config.temporal_network.hidden_channels
        self.time_dimension= config.temporal_network.time_embed_dim

        ##############################

        self.time_embed = Time_embedding(dim_time_emb=self.time_dimension, 
                                         dim_hidden=self.hidden_channels, 
                                         activation_fn=nn.ReLU())
        
        self.conv = TemporalGCNBlock(num_node_feat=self.num_node_features, 
                                     hidden_channels=self.hidden_channels, 
                                     time_channels=self.hidden_channels,
                                     out_channels=self.hidden_channels, 
                                     dropout=0.2)

        self.edge_encoding_0 = Linear(2*self.hidden_channels,self.hidden_channels)
        self.bn0_edge = nn.BatchNorm1d(self.hidden_channels)
        self.edge_encoding = Linear(self.hidden_channels + self.hidden_channels, 1)
        self.bn_edge = nn.BatchNorm1d(1)
        self.expected_output_shape = [self.number_of_nodes,self.number_of_nodes,1]
        self.mask_diagonal_off = torch.ones([self.number_of_nodes, self.number_of_nodes]) - torch.eye(self.number_of_nodes)
        self.mask_diagonal_off.unsqueeze_(0).unsqueeze_(-1)
        
    def forward(self, X, time):

        batch_size = X.size(0)
        x, edge_index, batch = sample_to_geometric(X,number_of_nodes=self.number_of_nodes)
        x = x.to(X.device)
        edge_index = edge_index.to(X.device)
        batch = batch.to(X.device)

        time_emb = self.time_embed(time)
        x = self.conv(x, time_emb, edge_index)

        x = torch.stack(unbatch(x,batch=batch),dim=0)
        N = self.number_of_nodes

        x_i = x.unsqueeze(2)  # Shape becomes (batch_size, N, 1, D)
        x_j = x.unsqueeze(1)  # Shape becomes (batch_size, 1, N, D)
        
        x = torch.cat((x_i.expand(-1, -1, N, -1), x_j.expand(-1, N, -1, -1)), dim=-1)  # Shape becomes (batch_size, N, N, 2*D)
        x = x.reshape(batch_size*N*N,2*self.hidden_channels)
        x = self.edge_encoding_0(x)  # Shape becomes (batch_size,N,N,D)
        x = self.bn0_edge(x)
        x = x.reshape(batch_size,N,N,self.hidden_channels)
        x = F.dropout(x, p=0.25, training=self.training)

        time_emb = time_emb.unsqueeze(1).unsqueeze(2).expand(-1, N, N, -1)  # Shape becomes (batch_size, N, N, time_emd_dim)

        x = torch.cat((x, time_emb), dim=-1)  # Shape becomes (batch_size, N, N, D + time_emd_dim)
        x = self.edge_encoding(x).squeeze()   # Shape becomes (batch_size, N, N, 1)
        x = x.reshape(batch_size*N*N,1)
        x = self.bn_edge(x)
        x = x.reshape(batch_size,N,N,1)
        x = F.dropout(x, p=0.25, training=self.training)
        
        self.mask_diagonal_off = self.mask_diagonal_off.to(X.device)
        x = x * self.mask_diagonal_off
        
        return x
    

class TemporalGCNBlock(nn.Module):
    def __init__( self, 
                 num_node_feat: int, 
                 hidden_channels: int, 
                 out_channels: int,
                 time_channels: int=None,
                 use_attention_block = False,
                 dropout: float=0.1):
        super().__init__()
        
        self.number_of_nodes = num_node_feat


        self.gcn_1 = GCNConv(2*num_node_feat, hidden_channels)
        self.bn_1 = BatchNorm(hidden_channels)

        self.gcn_2 = GCNConv(hidden_channels, hidden_channels)
        self.bn_2 = BatchNorm(hidden_channels)

        self.gcn_3 = GCNConv(hidden_channels, hidden_channels)
        self.bn_3 = BatchNorm(hidden_channels)

        self.time_emb = nn.Sequential(Linear(time_channels, num_node_feat),
                                      nn.BatchNorm1d(num_node_feat), 
                                      nn.ReLU(), 
                                      nn.Dropout(dropout)) 
        
        self.attention = NodeAttentionBlock(in_channels=2*hidden_channels, out_channels=hidden_channels) if use_attention_block else nn.Identity()
        self.initialize()

    def forward(self, x, t, edge_index):
        t = t.repeat_interleave(self.number_of_nodes,0)
        x = torch.cat([x, self.time_emb(t)], dim=-1)

        x1 = self.gcn_1(x, edge_index)
        x1 = self.bn_1(x1)
        x1 = F.relu(x1)

        x2 = self.gcn_2(x1, edge_index)
        x2 = self.bn_2(x2)
        x2 = F.relu(x2)     

        x3 = self.gcn_3(x2, edge_index)
        x3 = self.bn_3(x3)
        x3 = F.relu(x3)      
        x3 = self.attention(x3)
        return x3

    def initialize(self):
        for module in self.modules():
            if isinstance(module, Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)


class Time_embedding(nn.Module):
    def __init__(self, dim_hidden,  dim_time_emb, activation_fn=nn.ReLU()):
        super(Time_embedding, self).__init__()

        self.dim_time_emb = dim_time_emb

        layers = [ nn.Linear(dim_time_emb, dim_hidden),
                   activation_fn,
                   nn.Linear(dim_hidden, dim_hidden),
                  ]
        self.fc = nn.Sequential(*layers)
        self.initialize()

    def forward(self, t):
        temb = transformer_timestep_embedding(t.squeeze(), self.dim_time_emb, max_positions=10000)
        return self.fc(temb)
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1 
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class NodeAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Q = nn.Sequential(nn.Linear(in_channels, in_channels), BatchNorm1d(in_channels))
        self.K = nn.Sequential(nn.Linear(in_channels, in_channels), BatchNorm1d(in_channels))
        self.V = nn.Sequential(nn.Linear(in_channels, in_channels), BatchNorm1d(in_channels))
        self.proj = nn.Sequential(nn.Linear(in_channels, in_channels), 
                                  BatchNorm1d(in_channels),
                                  nn.ReLU(), 
                                  nn.Dropout(0.1),
                                  nn.Linear(in_channels, out_channels), 
                                  BatchNorm1d(out_channels),
                                  nn.Dropout(0.1))

    def forward(self, x):
        # x is of shape (batch_size, hidden_dim)
        batch_size, C = x.size()
        q = self.Q(x)  # (batch_size, C)
        k = self.K(x)  # (batch_size, C)
        v = self.V(x)  # (batch_size, C)
        q = q.view(batch_size, 1, C)  # (batch_size, 1, C)
        k = k.view(batch_size, 1, C)  # (batch_size, 1, C)
        v = v.view(batch_size, 1, C)  # (batch_size, 1, C)
        w = torch.bmm(q, k.transpose(1, 2)) * (C ** (-0.5))  # (batch_size, 1, 1)
        w = F.softmax(w, dim=-1)  # (batch_size, 1, 1)
        h = torch.bmm(w, v)  # (batch_size, 1, C)
        h = h.view(batch_size, C)  # (batch_size, C)
        h = self.proj(h)  # (batch_size, C)
        return h + x













# class EmbedFC(nn.Module):
#     def __init__(self, input_dim, emb_dim):
#         super(EmbedFC, self).__init__()
#         '''
#         generic one layer FC NN for embedding things  
#         '''
#         self.input_dim = input_dim
#         layers = [
#             nn.Linear(input_dim, emb_dim),
#             nn.GELU(),
#             nn.Linear(emb_dim, emb_dim),
#         ]
#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.view(-1, self.input_dim)
#         return self.model(x)
    
# class SimpleTemporalGCN(torch.nn.Module):
#     def __init__(self,config:CRMConfig):
#         super(SimpleTemporalGCN, self).__init__()
#         #torch.manual_seed(12345)

#         self.number_of_nodes = config.data1.max_node_num
#         self.num_node_features = self.number_of_nodes 
#         self.hidden_channels = config.temporal_network.hidden_channels
#         self.time_dimension= config.temporal_network.time_embed_dim

#         self.timeembed1 = Time_embedding(dim_time_emb=self.time_dimension, dim_hidden=self.time_dimension, activation_fn=nn.GELU())

#         self.conv1 = GCNConv(self.num_node_features, self.hidden_channels)
#         self.bn1 = BatchNorm(self.hidden_channels)

#         self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
#         self.bn2 = BatchNorm(self.hidden_channels)

#         self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels)
#         self.bn3 = BatchNorm(self.hidden_channels)

#         self.edge_encoding_0 = Linear(2*self.hidden_channels,self.hidden_channels)
#         self.bn0_edge = BatchNorm1d(self.hidden_channels)
#         self.edge_encoding = Linear(self.hidden_channels + self.time_dimension, 1)
#         self.bn_edge = BatchNorm1d(1)
#         self.expected_output_shape = [self.number_of_nodes,self.number_of_nodes,1]

#         self.mask_diagonal_off = torch.ones([self.number_of_nodes, self.number_of_nodes]) - torch.eye(self.number_of_nodes)
#         self.mask_diagonal_off.unsqueeze_(0)

#     def forward(self, X, time):
#         batch_size = X.size(0)
#         x, edge_index,batch = sample_to_geometric(X,number_of_nodes=self.number_of_nodes)
#         x = x.to(X.device)
#         edge_index = edge_index.to(X.device)
#         batch = batch.to(X.device)
#         time_emb = self.timeembed1(time)

#         # print(1, x.shape)
#         # 1. Obtain node embeddings 
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.dropout(x, p=0.2, training=self.training)

#         # print(2, x.shape)

#         # 2. Create outer concatenation for the edge encoding
#         x = torch.stack(unbatch(x,batch=batch),dim=0)
#         N = self.number_of_nodes

#         x_i = x.unsqueeze(2)  # Shape becomes (batch_size, N, 1, D)
#         x_j = x.unsqueeze(1)  # Shape becomes (batch_size, 1, N, D)
        
#         # Concatenate the expanded tensors along the last dimension
#         x = torch.cat((x_i.expand(-1, -1, N, -1), x_j.expand(-1, N, -1, -1)), dim=-1)  # Shape becomes (batch_size, N, N, 2*D)
#         x = x.reshape(batch_size*N*N,2*self.hidden_channels)
#         x = self.edge_encoding_0(x) # Shape becomes (batch_size,N,N,D)
#         x = self.bn0_edge(x)
#         x = x.reshape(batch_size,N,N,self.hidden_channels)
#         x = F.dropout(x, p=0.25, training=self.training)

#         # Expand time_emb to match the dimensions of B
#         time_emb = time_emb.unsqueeze(1).unsqueeze(2).expand(-1, N, N, -1)  # Shape becomes (batch_size, N, N, time_emd_dim)

#         # Concatenate time_emb_expanded to B along the last dimension
#         x = torch.cat((x, time_emb), dim=-1)  # Shape becomes (batch_size, N, N, D + time_emd_dim)
#         x = self.edge_encoding(x).squeeze() # Shape becomes (batch_size, N, N, 1)
#         x = x.reshape(batch_size*N*N,1)
#         x = self.bn_edge(x)
#         x = x.reshape(batch_size,N,N,1)
#         x = F.dropout(x, p=0.25, training=self.training)
        # self.mask_diagonal_off = self.mask_diagonal_off.to(X.device)
        # x = x * self.mask_diagonal_off
#         return x