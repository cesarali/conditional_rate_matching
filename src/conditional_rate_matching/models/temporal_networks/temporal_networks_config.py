from dataclasses import dataclass,asdict

@dataclass
class TemporalMLPConfig:
    name:str = "TemporalMLP"
    time_embed_dim :int = 39
    hidden_dim :int = 200