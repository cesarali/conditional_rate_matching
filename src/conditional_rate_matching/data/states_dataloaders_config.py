import os
from typing import List
from dataclasses import dataclass,asdict,field
from conditional_rate_matching import data_path

image_data_path = os.path.join(data_path,"raw")

@dataclass
class StatesDataloaderConfig:
    name:str = "StatesDataloader"
    dataset_name:str = "categorical_dirichlet" # categorical_dirichlet
    batch_size: int= 23
    data_dir:str = image_data_path

    max_test_size: int = None
    sample_size :int = None
    dirichlet_alpha :float = 100.
    bernoulli_probability:float = None

    dimensions: int = 4
    vocab_size: int = 2
    as_image: bool = False

    total_data_size: int = 60000
    training_size: int = 50000
    test_size: int = 10000
    test_split: float = 0.2

    temporal_net_expected_shape : List[int] = None
    data_min_max: List[float] = field(default_factory=lambda:[0.,1.])

    def __post_init__(self):
        self.temporal_net_expected_shape =  [self.dimensions]

