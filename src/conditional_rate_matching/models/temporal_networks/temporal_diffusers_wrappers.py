from torch import nn
from diffusers import UNet2DModel
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.models.temporal_networks.temporal_unet import UNet
from conditional_rate_matching.configs.configs_classes.config_crm import TemporalNetworkToRateConfig

class DiffusersUnetTau(nn.Module):

    def __init__(self,cfg:CRMConfig,device):
        super().__init__()

        in_chanels = cfg.temporal_network.input_channels

        self.expected_output_shape = [in_chanels,32,32]

        ch = cfg.temporal_network.ch
        num_res_blocks = cfg.temporal_network.num_res_blocks
        num_scales = cfg.temporal_network.num_scales
        ch_mult = cfg.temporal_network.ch_mult
        input_channels = cfg.temporal_network.input_channels
        output_channels = cfg.temporal_network.input_channels * cfg.data1.vocab_size
        scale_count_to_put_attn = cfg.temporal_network.scale_count_to_put_attn
        data_min_max = cfg.temporal_network.data_min_max
        dropout = cfg.temporal_network.dropout
        skip_rescale = cfg.temporal_network.skip_rescale
        do_time_embed = True
        time_scale_factor = cfg.temporal_network.time_scale_factor
        time_embed_dim = cfg.temporal_network.time_embed_dim

        self.temp_net = UNet(
            ch, num_res_blocks, num_scales, ch_mult, input_channels,
            output_channels, scale_count_to_put_attn, data_min_max,
            dropout, skip_rescale, do_time_embed, time_scale_factor,
            time_embed_dim
        ).to(device)

    def forward(self, x, times):
        return self.temp_net(x, times)

class DiffusersUnet2D(nn.Module):

    def __init__(self,config:CRMConfig,device):
        super().__init__()

        vocab_size = config.data1.vocab_size
        in_chanels = config.temporal_network.input_channels
        out_channels = in_chanels
        if isinstance(config.temporal_network_to_rate,TemporalNetworkToRateConfig):
            if config.temporal_network_to_rate.type_of == "empty":
                out_channels = in_chanels*vocab_size
            if config.temporal_network_to_rate.type_of == "logistic":
                out_channels = in_chanels*2
            
        vocab_size = config.data1.vocab_size

        self.temp_net = UNet2DModel(in_channels=in_chanels,
                                    out_channels=out_channels,
                                    norm_num_groups=32).to(device)

        self.expected_output_shape = [in_chanels*vocab_size,32,32]

    def forward(self, x, times):
        return self.temp_net(x, times).sample
