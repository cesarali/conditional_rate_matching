import os
from conditional_rate_matching import data_path
from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple, Dict
from conditional_rate_matching.configs.utils import expected_shape

graph_data_path = os.path.join(data_path,"raw","graph")

@dataclass
class AvailableGrayCodes:
    swissroll:str = "swissroll"
    circles: str = "circles"
    moons: str = "moons"
    gaussians: str = "gaussians"
    pinwheel: str = "pinwheel"
    spirals: str = "spirals"
    checkerboard: str = "checkerboard"
    line: str = "line"
    cos: str = "cos"

@dataclass
class GrayCodesDataloaderConfig:
    name:str = "GrayCodesDataloader"
    dataset_name:str ="swissroll"

    batch_size: int = 32

    discrete_dim:int = 32
    dimensions: int = 32
    vocab_size: int = 2

    total_data_size:int = None
    training_size:int = 160000
    test_size:int = 40000
    test_split: float = 0.2
    max_training_size:int = None
    max_test_size:int = None

    temporal_net_expected_shape : List[int] = field(default_factory=lambda:[32])
    data_min_max: List[float] = field(default_factory=lambda:[0.,1.])

    def __post_init__(self):
        self.total_data_size = self.training_size + self.test_size
        self.test_split = float(self.test_size)/float(self.total_data_size)
        self.training_proportion = 1. - self.test_split