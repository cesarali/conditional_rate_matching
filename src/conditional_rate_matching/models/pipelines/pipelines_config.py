from dataclasses import dataclass

@dataclass
class BasicPipelineConfig:
    name:str="BasicPipeline"
    number_of_steps:int = 20
    num_intermediates:int = 10
    time_epsilon = 1e-3

@dataclass
class DSBPipelineConfig:
    name:str="DSBPipeline"
    number_of_steps:int = 10
    num_intermediates:int = 10
    min_t = 0.1
    eps_ratio :float = 1e-9

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