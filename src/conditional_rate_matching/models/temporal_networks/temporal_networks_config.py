from typing import List,Tuple
from dataclasses import dataclass,asdict,field

@dataclass
class TemporalLeNet5Config:
    name : str = "TemporalLeNet5"
    time_embed_dim : int = 50
    hidden_dim : int = 84
    ema_decay: float = 0.999

@dataclass
class TemporalDeepMLPConfig:
    name : str = "TemporalDeepMLP"
    time_embed_dim : int = 50
    hidden_dim : int = 250
    activation : str = 'ReLU'
    num_layers : int = 4
    ema_decay: float = 0.999
    dropout : float = 0.2

@dataclass
class TemporalLeNet5Config:
    name : str = "TemporalLeNet5"
    time_embed_dim : int = 50
    hidden_dim : int = 84
    ema_decay: float = 0.999

@dataclass
class TemporalLeNet5AutoencoderConfig:
    name : str = "TemporalLeNet5Autoencoder"
    time_embed_dim : int = 128
    hidden_dim : int = 256
    ema_decay: float = 0.999

@dataclass
class TemporalUNetConfig:
    name : str = "TemporalUNet"
    time_embed_dim : int = 128
    hidden_dim : int = 256
    ema_decay: float = 0.999
    dropout : float = 0.1
    activation : str = 'GELU'


@dataclass
class TemporalGraphConvNetConfig:
    name : str = "TemporalGraphConvNet"
    time_embed_dim : int = 39
    hidden_dim : int = 200
    activation : str = 'ReLU'
    ema_decay: float = 0.999

@dataclass
class TemporalMLPConfig:
    name:str = "TemporalMLP"
    time_embed_dim :int = 100
    hidden_dim :int = 100
    ema_decay :float = 0.9999  # 0.9999

@dataclass
class ConvNetAutoencoderConfig:
    name: str = "ConvNetAutoencoder"
    ema_decay :float = 0.9999  # 0.9999

    encoder_channels: int = 16
    latent_dim: int = 32
    decoder_channels: int = 16

    time_embed_dim : int = 128
    time_scale_factor :int = 1000
    ema_decay :float = 0.9999  # 0.9999

@dataclass
class UConvNISTNetConfig:
    name: str = "UConvNISTNet"
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    time_embed_dim: int = 256
    ema_decay:float = 0.9999  # 0.9999


@dataclass
class DiffusersUnet2DConfig:
    name: str = "DiffusersUnet2D"
    num_res_blocks: int = 2
    num_scales: int = 4
    ch:int=128
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    input_channels: int = 1
    scale_count_to_put_attn: int = 1
    data_min_max: List[int] = field(default_factory=lambda: [0, 1])  # CHECK THIS for CIFAR 255
    dropout: float = 0.1
    skip_rescale: bool = True
    time_embed_dim: int = 128
    time_scale_factor: int = 1000
    ema_decay :float = 0.9999  # 0.9999

    def __post_init__(self):
        self.ch = self.time_embed_dim

NUM_CLASSES = 1000

@dataclass
class CFMUnetConfig:
    name: str =  "CFMUnet"
    dim: Tuple[int]= field(default_factory=lambda:(1, 28, 28))
    num_channels:int=32
    num_res_blocks:int=1
    channel_mult: int = None
    learn_sigma: int = False
    class_cond: int = False
    num_classes: int = NUM_CLASSES
    use_checkpoint: int = False
    attention_resolutions: int = "16"
    num_heads:int = 1
    num_head_channels:int = -1
    num_heads_upsample:int = -1
    use_scale_shift_norm:bool = False
    dropout:int = 0
    resblock_updown:bool = False
    use_fp16:bool = False
    use_new_attention_order:bool = False
    ema_decay:float = 0.999

@dataclass
class TemporalScoreNetworkAConfig:
    name: str = "TemporalScoreNetworkA"
    conv: str = "GCN" # MLP,GCN
    num_heads:int = 4
    depth: int = 3
    adim: int = 32
    nhid: int = 32
    num_layers: int = 5
    num_linears: int = 2
    c_init: int = 2
    c_hid: int = 8
    c_final: int = 4

    dropout_rate:float = 0.1
    use_bn: bool = False

    time_embed_dim: int = 128
    time_scale_factor: int = 1000

    ema_decay :float = 0.9999  # 0.9999


@dataclass
class SequenceTransformerConfig:
    name: str = "SequenceTransformer"
    num_layers:int = 6
    d_model:int = 128
    num_heads:int = 8
    dim_feedforward:int = 2048
    dropout:float = 0.1
    temb_dim:int = 128
    num_output_FFresiduals:int = 2
    time_scale_factor:int = 1000
    use_one_hot_input:bool = True

    ema_decay :float = 0.9999  # 0.9999

@dataclass
class SimpleTemporalGCNConfig:
    name:str = "SimpleTemporalGCN"
    time_embed_dim:int = 19
    hidden_channels:int = 64
    ema_decay :float = 0.9999  # 0.9999


