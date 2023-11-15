from dataclasses import dataclass,asdict

@dataclass
class TemporalMLPConfig:
    name:str = "TemporalMLP"
    time_embed_dim :int = 39
    hidden_dim :int = 200

@dataclass
class ConvNetAutoencoderConfig:
    temp_name: str = "ConvNetAutoencoder"

    encoder_channels: int = 16
    latent_dim: int = 32
    decoder_channels: int = 16

    time_embed_dim : int = 128
    time_scale_factor :int = 1000
