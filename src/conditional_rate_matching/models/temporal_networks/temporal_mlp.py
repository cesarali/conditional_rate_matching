import torch
import numpy as np
from torch import nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn.functional as F
from conditional_rate_matching.utils.activations import get_activation_function
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig as CRMConfig
from conditional_rate_matching.models.temporal_networks.temporal_embedding_utils import transformer_timestep_embedding

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
        self.dropout_rate = config.temporal_network.dropout  # Assuming dropout rate is specified in the config

        layers = [nn.Linear(self.dimensions + self.time_embed_dim, self.hidden_layer),
                  nn.BatchNorm1d(self.hidden_layer),
                  self.act_fn]

        if self.dropout_rate: layers.append(nn.Dropout(self.dropout_rate))  # Adding dropout if specified

        for _ in range(self.num_layers - 2):
            layers.extend([nn.Linear(self.hidden_layer, self.hidden_layer),
                           nn.BatchNorm1d(self.hidden_layer),
                           self.act_fn])
            if self.dropout_rate: layers.extend([nn.Dropout(self.dropout_rate)])  # Adding dropout

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


class TemporalLeNet5(nn.Module):

    def __init__(self,
                 config,
                 device):
<<<<<<< HEAD

=======
>>>>>>> origin/main
        super().__init__()
        self.dimensions = config.data0.dimensions
        self.vocab_size = config.data0.vocab_size
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.define_deep_models()
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [28, 28, self.vocab_size]

    def define_deep_models(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn2d1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2d2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4 + self.time_embed_dim, self.hidden_layer)
        self.bn1 = nn.BatchNorm1d(self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer + self.time_embed_dim, self.hidden_layer)
        self.bn2 = nn.BatchNorm1d(self.hidden_layer)
        self.fc3 = nn.Linear(self.hidden_layer + self.time_embed_dim, 28 * 28 * 2)

<<<<<<< HEAD

=======
>>>>>>> origin/main
    def forward(self, x, times):
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)

        x = F.max_pool2d(F.relu(self.bn2d1(self.conv1(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2d2(self.conv2(x))), (2, 2))
        x = x.view(-1, np.prod(x.size()[1:]))
<<<<<<< HEAD
        
=======

>>>>>>> origin/main
        x = torch.concat([x, time_embeddings], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))

        x = torch.concat([x, time_embeddings], dim=1)
        x = F.relu(self.bn2(self.fc2(x)))

        x = torch.concat([x, time_embeddings], dim=1)
        x = self.fc3(x)

<<<<<<< HEAD
        return x.view(-1, 28, 28, 2) 
=======
        return x.view(-1, 28, 28, 2)
>>>>>>> origin/main

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


<<<<<<< HEAD
class TemporalLeNet5Autoencoder(nn.Module):

    def __init__(self,
                 config,
                 device):

        super().__init__()
        self.dimensions = config.data0.dimensions
        self.vocab_size = config.data0.vocab_size
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_layer = config.temporal_network.hidden_dim
        self.Encoder()
        self.Bottleneck()
        self.Decoder()
        self.init_weights()
        self.to(device)
        self.expected_output_shape = [28, 28, self.vocab_size]

    def Encoder(self):
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4 + self.time_embed_dim, 120)
        self.bn1 = nn.BatchNorm1d(120)

    def Bottleneck(self):
        self.bottleneck = nn.Linear(120 + self.time_embed_dim, self.hidden_layer) 
        self.bnb = nn.BatchNorm1d(self.hidden_layer)

    def Decoder(self):
        self.fc2 = nn.Linear(self.hidden_layer, 120)
        self.bn2 = nn.BatchNorm1d(120)
        self.fc3 = nn.Linear(120, 16 * 4 * 4)
        self.bn3 = nn.BatchNorm1d(16 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(16, 6, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(6, 2, 18, stride=1)

    def forward(self, x, times):
        time_embeddings = transformer_timestep_embedding(times, embedding_dim=self.time_embed_dim)
        # Encoder
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, np.prod(x.size()[1:]))
        x = torch.concat([x, time_embeddings], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))

        # Bottleneck
        x = torch.concat([x, time_embeddings], dim=1)
        x = F.relu(self.bnb(self.bottleneck(x)))

        # Decoder
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = x.view(-1, 16, 4, 4)  # Reshape to match the feature map size
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        rate_logits  = x.permute(0, 2, 3, 1)
        return rate_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.bottleneck.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)


class TemporalUNet(nn.Module):

    def __init__(self,
                 config,
                 device):

        super().__init__()
        self.dimensions = config.data0.dimensions
        self.vocab_size = config.data0.vocab_size
        self.time_embed_dim = config.temporal_network.time_embed_dim
        self.hidden_dim = config.temporal_network.hidden_dim
        self.Encoder()
        self.TimeEmbedding()
        self.Decoder()
        self.to(device)
        self.expected_output_shape = [28, 28, self.vocab_size]

    def Encoder(self):
        self.init_conv = ResidualConvBlock(1, self.hidden_dim , is_res=True)
        self.down1 = UnetDown(self.hidden_dim, self.hidden_dim)
        self.down2 = UnetDown(self.hidden_dim, 2 * self.hidden_dim)
        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

    def TimeEmbedding(self):
        self.timeembed1 = EmbedFC(1, 2*self.hidden_dim)
        self.timeembed2 = EmbedFC(1, 1*self.hidden_dim)

    def Decoder(self):
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.hidden_dim, 2 * self.hidden_dim, 7, 7), 
            nn.GroupNorm(8, 2 * self.hidden_dim),
            nn.GELU())

        self.up1 = UnetUp(4 * self.hidden_dim, self.hidden_dim)
        self.up2 = UnetUp(2 * self.hidden_dim, self.hidden_dim)
        self.out = nn.Sequential(
            nn.Conv2d(2 * self.hidden_dim, self.hidden_dim, 3, 1, 1),
            nn.GroupNorm(8, self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.vocab_size, 3, 1, 1),
        )

    def forward(self, x, times):
        x = self.init_conv(x)
        
        # encode:
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed:
        temb1 = self.timeembed1(times).view(-1, self.hidden_dim * 2, 1, 1)
        temb2 = self.timeembed2(times).view(-1, self.hidden_dim, 1, 1)
        
        # decode:
        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2) 
        up3 = self.up2(up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))

        return out.permute(0, 2, 3, 1) 


class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


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
=======

>>>>>>> origin/main

class TemporalMLP(nn.Module):
    """
    """
    def __init__(self, config:CRMConfig, device):
        super().__init__()
        if hasattr(config,'data1'):
            config_data = config.data1
        else:
            config_data = config.data0

        self.dimensions = config_data.dimensions
        self.vocab_size = config_data.vocab_size
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
