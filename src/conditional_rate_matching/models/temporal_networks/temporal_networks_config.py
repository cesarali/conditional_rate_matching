from dataclasses import dataclass,asdict


@dataclass
class TemporalDeepMLPConfig:
    name : str = "TemporalDeepMLP"
    time_embed_dim : int = 39
    hidden_dim : int = 200
    activation : str = 'ReLU'
    num_layers : int = 2

@dataclass
class TemporalDeepSetsConfig:
    name : str = "TemporalDeepSets"
    time_embed_dim : int = 39
    hidden_dim : int = 200
    pool : str = "sum"
    activation : str = 'ReLU'
    num_layers : int = 2

@dataclass
class TemporalGraphConvNetConfig:
    name : str = "TemporalGraphConvNet"
    time_embed_dim : int = 39
    hidden_dim : int = 200
    activation : str = 'ReLU'

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
