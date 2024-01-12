
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import math
from typing import Any
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

# -------- GCN layer --------
class TemporalDenseGCNConv(torch.nn.Module):
    r"""See :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels, out_channels,improved=False, bias=True,time_embed_dim=19):
        super(TemporalDenseGCNConv, self).__init__()
        self.time_embed_dim = time_embed_dim

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

        self.time_weight = torch.nn.Linear(self.time_embed_dim, out_channels)
        self.time_bias = torch.nn.Linear(self.time_embed_dim, out_channels)


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, x, adj, time, mask=None, add_loop=True):
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
                \times N \times F}`, with batch-size :math:`B`, (maximum)
                number of nodes :math:`N` for each graph, and feature
                dimension :math:`F`.
            adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
                \times N \times N}`. The adjacency tensor is broadcastable in
                the batch dimension, resulting in a shared adjacency matrix for
                the complete batch.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            add_loop (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
        time_embeddings = transformer_timestep_embedding(time, embedding_dim=self.time_embed_dim)
        time_weight_ = self.time_weight(time_embeddings)
        time_bias_ = self.time_bias(time_embeddings)

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1 if not self.improved else 2

        out = torch.matmul(x, self.weight)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)
        out = time_weight_[:, None, :] * out
        if self.bias is not None:
            out = out + self.bias + time_bias_[:,None,:]

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

# -------- MLP layer --------

class MLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            num_classes: the number of classes of input, to be treated with different gains and biases,
                    (see the definition of class `ConditionalLayer1d`)
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.activate_func = activate_func

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = torch.nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

            if self.use_bn:
                self.batch_norms = torch.nn.ModuleList()
                for layer in range(num_layers - 1):
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))


    def forward(self, x):
        """
        :param x: [num_classes * batch_size, N, F_i], batch of node features
            note that in self.cond_layers[layer],
            `x` is splited into `num_classes` groups in dim=0,
            and then treated with different gains and biases
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                if self.use_bn:
                    h = self.batch_norms[layer](h)
                h = self.activate_func(h)
            return self.linears[self.num_layers - 1](h)

class TemporalLinear(torch.nn.Module):

    def __init__(self,input_dim,output_dim,time_embed_dim=19):
        super().__init__()
        self.StaticLinear = torch.nn.Linear(input_dim,output_dim)
        self.TimeTransformationWeight = torch.nn.Linear(time_embed_dim, output_dim)
        self.TimeTransformationBias = torch.nn.Linear(time_embed_dim, output_dim)
        self.time_embed_dim = time_embed_dim

    def forward(self,x,t):
        time_emb = transformer_timestep_embedding(t,embedding_dim=self.time_embed_dim)
        weight_time = self.TimeTransformationWeight(time_emb)
        bias_time = self.TimeTransformationBias(time_emb)
        out = self.StaticLinear(x)
        out = out*weight_time[:,None,None,:] + bias_time[:,None,None,:]
        return out

class TemporalMLP(torch.nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=F.relu,temp_dim=19):
        """
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            num_classes: the number of classes of input, to be treated with different gains and biases,
                    (see the definition of class `ConditionalLayer1d`)
        """

        super(TemporalMLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.use_bn = use_bn
        self.activate_func = activate_func

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = TemporalLinear(input_dim, output_dim,temp_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(TemporalLinear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(TemporalLinear(hidden_dim, hidden_dim))
            self.linears.append(TemporalLinear(hidden_dim, output_dim))

            if self.use_bn:
                self.batch_norms = torch.nn.ModuleList()
                for layer in range(num_layers - 1):
                    self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))


    def forward(self, x,time):
        """
        :param x: [num_classes * batch_size, N, F_i], batch of node features
            note that in self.cond_layers[layer],
            `x` is splited into `num_classes` groups in dim=0,
            and then treated with different gains and biases
        """
        if self.linear_or_not:
            # If linear model
            return self.linear(x,time)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h,time)
                if self.use_bn:
                    h = self.batch_norms[layer](h)
                h = self.activate_func(h)
            return self.linears[self.num_layers - 1](h,time)

