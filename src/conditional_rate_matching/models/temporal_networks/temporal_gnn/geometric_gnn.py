
import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)
    
class SimpleTemporalGCN(torch.nn.Module):
    def __init__(self,config:CRMConfig):
        super(SimpleTemporalGCN, self).__init__()
        #torch.manual_seed(12345)

        self.number_of_nodes = config.data1.max_node_num
        self.num_node_features = self.number_of_nodes 
        self.hidden_channels = config.temporal_network.hidden_channels
        self.time_dimension= config.temporal_network.time_embed_dim

        self.conv1 = GCNConv(self.num_node_features, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels)

        self.timeembed1 = EmbedFC(1, self.time_dimension)

        self.edge_encoding_0 = Linear(2*self.hidden_channels,self.hidden_channels)
        self.edge_encoding = Linear(self.hidden_channels+self.time_dimension,1)

        self.expected_output_shape = [self.number_of_nodes,self.number_of_nodes,1]

    def forward(self, X, time):
        batch_size = X.size(0)
        x,edge_index,batch = sample_to_geometric(X,number_of_nodes=self.number_of_nodes)
        time_emb = self.timeembed1(time)

        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)

        # 2. Create outer concatenation for the edge encoding
        x = torch.stack(unbatch(x,batch=batch),dim=0)
        N = self.number_of_nodes

        x_i = x.unsqueeze(2)  # Shape becomes (batch_size, N, 1, D)
        x_j = x.unsqueeze(1)  # Shape becomes (batch_size, 1, N, D)
        
        # Concatenate the expanded tensors along the last dimension
        x = torch.cat((x_i.expand(-1, -1, N, -1), x_j.expand(-1, N, -1, -1)), dim=-1)  # Shape becomes (batch_size, N, N, 2*D)
        x = self.edge_encoding_0(x) # Shape becomes (batch_size,N,N,D)

        # Expand time_emb to match the dimensions of B
        time_emb = time_emb.unsqueeze(1).unsqueeze(2).expand(-1, N, N, -1)  # Shape becomes (batch_size, N, N, time_emd_dim)

        # Concatenate time_emb_expanded to B along the last dimension
        x = torch.cat((x, time_emb), dim=-1)  # Shape becomes (batch_size, N, N, D + time_emd_dim)
        x = self.edge_encoding(x).squeeze() # Shape becomes (batch_size, N, N,1)

        return x