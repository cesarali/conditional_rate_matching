from dataclasses import dataclass

@dataclass
class OopsPipelineConfig:
    num_samples:int = 16
    dynamic_binarization:bool = False
    input_type:str = "binary"
    buffer_init: str = "data" # uniform
    buffer_size: int = 1000
    number_of_betas: int = 100 #10000
    sampler_steps_per_ais_iter:int = 1
    viz_every:int = 10
    reinit_freq:float = 0.0