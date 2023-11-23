import torch
from torch import nn
from conditional_rate_matching.models.temporal_networks.embedding_utils import transformer_timestep_embedding
from conditional_rate_matching.configs.config_crm import CRMConfig
from conditional_rate_matching.configs.config_crm import CRMConfig as CRMConfig

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

class TemporalMLP(nn.Module):
    """
    Simple 2-layer MLP with no activations
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


# Define the Autoencoder class
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
