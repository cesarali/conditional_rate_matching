from dataclasses import dataclass

@dataclass
class OopsPipelineConfig:
    num_gibbs_steps:int = 100
    num_samples:int = 16
    dynamic_binarization:bool = False
    input_type:str = "binary"
    buffer_init: str = "data" # uniform
    buffer_size: int = 1000
    n_iters:int = 100 #10000
    n_samples:int = 100 #test_batch_size
    steps_per_iter:int = 1
    viz_every:int = 100
    reinit_freq:float = 0.0