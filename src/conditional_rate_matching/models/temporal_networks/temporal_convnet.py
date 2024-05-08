import torch
from torch import nn as nn
from conditional_rate_matching.models.networks.mlp import ResNetBlock
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig as CRMConfig
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding
from conditional_rate_matching.models.generative_models.ddsm import GaussianFourierProjection

def binary_to_onehot(x):
    xonehot = []
    xonehot.append((x == 1)[..., None])
    xonehot.append((x == 0)[..., None])
    return torch.cat(xonehot, -1)

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

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]

def x_sigmoid(x):
    return x * torch.sigmoid(x)

class UConvNISTNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, config:CRMConfig):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """

        channels = config.temporal_network.channels
        embed_dim = config.temporal_network.time_embed_dim
        self.expected_output_shape = [28, 28,2]

        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(2, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3], channels[2], 3, stride=2, bias=False
        )

        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2],
            channels[1],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )

        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1],
            channels[0],
            3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 2, 3, stride=1)

        # The swish activation function
        self.act = x_sigmoid

        self.to_pre_rate =  nn.Linear(28*28,28*28*2)

    def forward(self, x, t):
        if x.size(1) == 1:
            x = x.squeeze()
            x = x.view(-1,28*28)
            x = self.to_pre_rate(x)
            x = x.view(-1,28,28,2)

        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t / 4.0))
        # Encoding path
        h1 = self.conv1(x.permute(0, 3, 1, 2))
        ## Incorporate information from t
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # some scaling here by time or x may be helpful but no scaling works fine here
        h = h.permute(0, 2, 3, 1)
        h = h - h.mean(axis=-1, keepdims=True)
        return h