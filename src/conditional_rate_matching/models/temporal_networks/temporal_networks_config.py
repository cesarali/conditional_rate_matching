from typing import List
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
class TemporalDeepSetsConfig:
    name : str = "TemporalDeepSets"
    time_embed_dim : int = 39
    hidden_dim : int = 200
    pool : str = "sum"
    activation : str = 'ReLU'
    num_layers : int = 2
    ema_decay: float = 0.999

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
    ch_mult: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    input_channels: int = 1
    scale_count_to_put_attn: int = 1
    data_min_max: List[int] = field(default_factory=lambda: [0, 1])  # CHECK THIS for CIFAR 255
    dropout: float = 0.1
    skip_rescale: bool = True
    time_embed_dim: int = 128
    time_scale_factor: int = 1000
    ema_decay :float = 0.9999  # 0.9999


@dataclass
class TemporalScoreNetworkAConfig:
    name: "str" = "TemporalScoreNetworkA"
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

    time_embed_dim: int = 128
    time_scale_factor: int = 1000

    ema_decay :float = 0.9999  # 0.9999

