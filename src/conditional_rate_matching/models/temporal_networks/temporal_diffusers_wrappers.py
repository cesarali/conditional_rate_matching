import torch
from torch import nn
from diffusers import UNet2DModel
from conditional_rate_matching.configs.config_crm import CRMConfig
from conditional_rate_matching.configs.config_ctdd import CTDDConfig

from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig,DiscreteCIFAR10Config

class DiffusersUnet2D(nn.Module):

    def __init__(self,config:CRMConfig,device):
        super().__init__()

        in_chanels = config.temporal_network.input_channels
        self.temp_net = UNet2DModel(in_channels=in_chanels,
                                    out_channels=in_chanels,
                                    norm_num_groups=32).to(device)

        self.expected_output_shape = [in_chanels,32,32]

        #ch = cfg.temp_network.ch
        #num_res_blocks = cfg.temp_network.num_res_blocks
        #num_scales = cfg.temp_network.num_scales
        #ch_mult = cfg.temp_network.ch_mult
        #input_channels = cfg.temp_network.input_channels
        #output_channels = cfg.temp_network.input_channels * cfg.data.S
        #scale_count_to_put_attn = cfg.temp_network.scale_count_to_put_attn
        #data_min_max = cfg.temp_network.data_min_max
        #dropout = cfg.temp_network.dropout
        #skip_rescale = cfg.temp_network.skip_rescale
        #do_time_embed = True
        #time_scale_factor = cfg.temp_network.time_scale_factor
        #time_embed_dim = cfg.temp_network.time_embed_dim

        #self.temp_net = networks_tau.UNet(
        #    ch, num_res_blocks, num_scales, ch_mult, input_channels,
        #    output_channels, scale_count_to_put_attn, data_min_max,
        #    dropout, skip_rescale, do_time_embed, time_scale_factor,
        #    time_embed_dim
        #).to(device)

    def forward(self, x, times):
        return self.temp_net(x, times).sample
