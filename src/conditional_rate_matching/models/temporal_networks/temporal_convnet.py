from torch import nn as nn
from conditional_rate_matching.models.networks.mlp import ResNetBlock
from conditional_rate_matching.configs.config_crm import CRMConfig as CRMConfig
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding


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
