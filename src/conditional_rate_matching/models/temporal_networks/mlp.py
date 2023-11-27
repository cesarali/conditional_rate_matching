from platform import node
import time
import torch
import torch.nn as nn
from conditional_rate_matching.configs.config_crm import CRMConfig
from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.configs.config_crm import CRMConfig as CRMConfig
from conditional_rate_matching.models.temporal_networks.embedding_utils import transformer_timestep_embedding

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

def mlp_ebm(nin, nint=256, nout=1):
    return nn.Sequential(
        nn.Linear(nin, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nint),
        Swish(),
        nn.Linear(nint, nout),
    )


class MLPEBM_cat(nn.Module):


    def __init__(self, nin, n_proj, n_cat=256, nint=256, nout=1):
        super().__init__()
        self.proj = nn.Linear(n_cat, n_proj)
        self.n_proj = n_proj
        self.net = mlp_ebm(nin * n_proj, nint, nout=nout)

    def forward(self, x):
        xr = x.view(x.size(0) * x.size(1), x.size(2))
        xr_p = self.proj(xr)
        x_p = xr_p.view(x.size(0), x.size(1), self.n_proj)
        x_p = x_p.view(x.size(0), x.size(1) * self.n_proj)
        return self.net(x_p)


class MLP_EBM(nn.Module):

    def __init__(self,config:OopsConfig):
        super().__init__()
        dimensions = config.data0.dimensions
        nint = config.model_mlp.hidden_size
        self.net = mlp_ebm(nin=dimensions, nint=nint, nout=1)

    def forward(self, x):
        return self.net(x).squeeze()

def conv_transpose_3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=3, stride=stride, padding=1, output_padding=1, bias=True)


def conv3x3(in_planes, out_planes, stride=1):
    if stride < 0:
        return conv_transpose_3x3(in_planes, out_planes, stride=-stride)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, out_nonlin=True):
        super(BasicBlock, self).__init__()
        self.nonlin1 = Swish()
        self.nonlin2 = Swish()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.out_nonlin = out_nonlin

        self.shortcut_conv = None
        if stride != 1 or in_planes != self.expansion * planes:
            if stride < 0:
                self.shortcut_conv = nn.ConvTranspose2d(in_planes, self.expansion * planes,
                                                        kernel_size=1, stride=-stride,
                                                        output_padding=1, bias=True)
            else:
                self.shortcut_conv = nn.Conv2d(in_planes, self.expansion * planes,
                                               kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        out = self.nonlin1(self.conv1(x))
        out = self.conv2(out)
        if self.shortcut_conv is not None:
            out_sc = self.shortcut_conv(x)
            out += out_sc
        else:
            out += x
        if self.out_nonlin:
            out = self.nonlin2(out)
        return out


class ResNetEBM(nn.Module):
    def __init__(self,config:OopsConfig):
        n_channels = config.model_mlp.n_channels
        n_blocks = config.model_mlp.n_blocks
        super().__init__()
        self.proj = nn.Conv2d(1, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(n_blocks)]
        all = downsample + main
        self.net = nn.Sequential(*all)
        self.energy_linear = nn.Linear(n_channels, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        input = self.proj(input)
        out = self.net(input)
        out = out.view(out.size(0), out.size(1), -1).mean(-1)
        return self.energy_linear(out).squeeze()


class MNISTConvNet(nn.Module):
    def __init__(self, nc=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, nc, 3, 1, 1),
            Swish(),
            nn.Conv2d(nc, nc * 2, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc * 2, nc * 2, 3, 1, 1),
            Swish(),
            nn.Conv2d(nc * 2, nc * 4, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc * 4, nc * 4, 3, 1, 1),
            Swish(),
            nn.Conv2d(nc * 4, nc * 8, 4, 2, 1),
            Swish(),
            nn.Conv2d(nc * 8, nc * 8, 3, 1, 0),
            Swish(),
        )
        self.out = nn.Linear(nc * 8, 1)

    def forward(self, input):
        input = input.view(input.size(0), 1, 28, 28)
        out = self.net(input)
        out = out.squeeze()
        return self.out(out).squeeze()


class ResNetEBM_cat(nn.Module):
    def __init__(self, shape, n_proj, n_cat=256, n_channels=64):
        super().__init__()
        self.shape = shape
        self.n_cat = n_cat
        self.proj = nn.Conv2d(n_cat, n_proj, 1, 1, 0)
        self.proj2 = nn.Conv2d(n_proj, n_channels, 3, 1, 1)
        downsample = [
            BasicBlock(n_channels, n_channels, 2),
            BasicBlock(n_channels, n_channels, 2)
        ]
        main = [BasicBlock(n_channels, n_channels, 1) for _ in range(6)]
        all = downsample + main
        self.net = nn.Sequential(*all)
        self.energy_linear = nn.Linear(n_channels, 1)

    def forward(self, input):
        input = input.view(input.size(0), self.shape[1], self.shape[2], self.n_cat).permute(0, 3, 1, 2)
        input = self.proj(input)
        input = self.proj2(input)
        out = self.net(input)
        out = out.view(out.size(0), out.size(1), -1).mean(-1)
        return self.energy_linear(out).squeeze()

class TemporalMLP(nn.Module):
    """

    """
    def __init__(self, config:CRMConfig, device):
        super().__init__()
        self.dimensions = config.data0.dimensions
        self.vocab_size = config.data0.vocab_size
        self.define_deep_models(config)
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [self.dimensions,self.vocab_size]

    def define_deep_models(self,config):
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.f1 = nn.Linear(self.dimensions, self.hidden_layer)
        self.f2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.dimensions * self.vocab_size)

    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embbedings = transformer_timestep_embedding(times,
                                                         embedding_dim=self.time_embed_dim)

        step_one = self.f1(x)
        step_two = torch.concat([step_one, time_embbedings], dim=1)
        rate_logits = self.f2(step_two)
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)

        return rate_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.f1.weight)
        nn.init.xavier_uniform_(self.f2.weight)

# Define a ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temb_dim=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_dim is not None:
            self.act = nn.functional.silu
            self.dense0 = nn.Linear(temb_dim, out_channels)
            nn.init.zeros_(self.dense0.bias)

    def forward(self, x, temb=None):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x += residual  # Add the residual connection
        #x = self.relu(x)

        if temb is not None:
            h = self.dense0(self.act(temb))
            x+= h[:,:,None,None]
        return x

class ConvNetAutoencoder(nn.Module):
    def __init__(self, config:CRMConfig,device):
        super(ConvNetAutoencoder, self).__init__()

        self.device = device

        self.in_channels = 1

        self.encoder_channels = config.temporal_network.encoder_channels
        self.latent_dim = config.temporal_network.latent_dim
        self.decoder_channels = config.temporal_network.encoder_channels

        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.time_scale_factor = config.temporal_network.time_scale_factor

        self.expected_output_shape = [2*self.in_channels,28,28]

        self.do_time_embed = True
        self.act = nn.functional.silu

        # time
        self.temb_modules = []
        self.temb_modules.append(nn.Linear(self.time_embed_dim, self.time_embed_dim*4))
        nn.init.zeros_(self.temb_modules[-1].bias)
        self.temb_modules.append(nn.Linear(self.time_embed_dim*4, self.time_embed_dim*4))
        nn.init.zeros_(self.temb_modules[-1].bias)
        self.temb_modules = nn.ModuleList(self.temb_modules).to(self.device)
        self.expanded_time_dim = 4 * self.time_embed_dim if self.do_time_embed else None

        # Encoder
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

        self.resnet_1 = ResNetBlock(32, 32, self.expanded_time_dim)  # Use the ResNet block

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResNetBlock(64, 64,self.expanded_time_dim),  # Use the ResNet block
        )

        # Decoder
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.resnet_2 = ResNetBlock(32, 32,self.expanded_time_dim)  # Use the ResNet block

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(32, 2*self.in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.to(self.device)

    def _time_embedding(self, timesteps):
        if self.do_time_embed:
            temb = transformer_timestep_embedding(
                timesteps * self.time_scale_factor, self.time_embed_dim
            )
            temb = self.temb_modules[0](temb)
            temb = self.temb_modules[1](self.act(temb))
        else:
            temb = None
        return temb

    def forward(self,x,timesteps):
        if len(x.shape) == 2:
            x = x.view(-1,1,28,28)
        temp = self._time_embedding(timesteps)

        # Encoder
        x = self.encoder_1(x)
        x = self.resnet_1(x,temp)
        x = self.encoder_2(x)

        # Decoder
        x = self.decoder_1(x)
        x = self.resnet_2(x,temp)
        x = self.decoder_2(x)

        return x
    

### MY ARCHITECTURES ###


def get_activation_function(name: str='ReLU'):
    if name is not None:
        activation_functions = {"ReLU": nn.ReLU(),
                                "LeakyReLU": nn.LeakyReLU(),
                                "ELU": nn.ELU(),
                                "SELU": nn.SELU(),
                                "GLU": nn.GLU(),
                                "GELU": nn.GELU(),
                                "CELU": nn.CELU(),
                                "PReLU": nn.PReLU(),
                                "Sigmoid": nn.Sigmoid(),
                                "Tanh": nn.Tanh(),
                                "Hardswish": nn.Hardswish(),
                                "Hardtanh": nn.Hardtanh(),
                                "LogSigmoid": nn.LogSigmoid(),
                                "Softplus": nn.Softplus(),
                                "Softsign": nn.Softsign(),
                                "Softshrink": nn.Softshrink(),
                                "Softmin": nn.Softmin(),
                                "Softmax": nn.Softmax()}
        return activation_functions[name]
    else: return None


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
        self.act_fn = get_activation_function(config.temporal_network.activation)

        layers = [nn.Linear(self.dimensions + self.time_embed_dim, self.hidden_layer)]
        if self.act_fn: layers.append(self.act_fn)
        
        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_layer, self.hidden_layer)])
            if self.act_fn: layers.extend([self.act_fn])

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


class TemporalDeepSets(torch.nn.Module):
    def __init__(self, 
                 config, 
                 device,
                 pool='meansum'
                 ):
        super().__init__()
        self.dimensions = int(np.sqrt(config.data0.dimensions))  # dxd -> d
        self.vocab_size = config.data0.vocab_size
        self.define_deep_models(config)
        self.to(device)
        self.expected_output_shape = [self.dimensions, self.vocab_size]
    
    
    def define_deep_models(self, config):
        self.dim_hidden_t = config.temporal_network.time_embed_dim
        self.dim_hidden_x = config.temporal_network.hidden_dim
        self.num_layers = config.temporal_network.num_layers
        self.act_fn = get_activation_function(config.temporal_network.activation)
        self.pool = config.temporal_network.pool

        s = 2 if self.pool == 'meansum' else 1  

        phi_layers = [torch.nn.Linear(self.dimensions + self.dim_hidden_t, self.dim_hidden_x), self.act_fn]
        for _ in range(self.num_layers-1): phi_layers.extend([torch.nn.Linear(self.dim_hidden_x, self.dim_hidden_x), self.act_fn])
        phi_layers.append(torch.nn.Linear(self.dim_hidden_x, self.dim_hidden_x))
        self.phi = torch.nn.Sequential(*phi_layers)

        rho_layers = [torch.nn.Linear(s * self.dim_hidden_x, self.dim_hidden_x), self.act_fn]
        for _ in range(self.num_layers-1): rho_layers.extend([torch.nn.Linear(self.dim_hidden_x, self.dim_hidden_x), self.act_fn])
        rho_layers.append(torch.nn.Linear(self.dim_hidden_x, self.dimensions * self.vocab_size))
        self.rho = torch.nn.Sequential(*rho_layers)

    def forward(self, x, times):
        batch_size = x.shape[0]
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.dim_hidden_t)
        time_embeddings = time_embeddings.unsqueeze(1).repeat(1, x.shape[1], 1) 
        x = torch.concat([x, time_embeddings], dim=-1)
        
        #...deepsets:
        h = self.phi(x)
        h_sum = h.sum(1, keepdim=False)   
        h_mean = h.mean(1, keepdim=False)  
        
        #...aggregation:
        if self.pool == 'sum': h_pool = h_sum  
        elif self.pool == 'mean': h_pool = h_mean 
        elif self.pool == 'meansum': h_pool = torch.cat([h_mean, h_sum], dim=1) 

        rate_logits = self.rho(h_pool)    
        rate_logits = rate_logits.reshape(batch_size, self.dimensions, self.vocab_size)
        return rate_logits
    
import torch_geometric
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

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

        #...define GNN layers
        self.conv1 = GCNConv(self.dim_hidden_t + 1, self.dim_hidden_x)
        self.conv2 = GCNConv(self.dim_hidden_x, self.dim_hidden_x)
        self.linear = nn.Linear(self.dim_hidden_x, self.dimensions * self.vocab_size )

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
