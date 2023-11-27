import os
from conditional_rate_matching import data_path
from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple, Dict
from conditional_rate_matching.configs.utils import expected_shape

graph_data_path = os.path.join(data_path,"raw","gray_code")

@dataclass
class GrayCodeDataloaderConfig:
    name:str = "GrayCodeDataloader"
    dataset_name: str =None
    batch_size: int=None
    data_dir:str = graph_data_path

    dimensions: int = None
    vocab_size: int = 2

    flatten: bool = True
    as_image: bool= True

    total_data_size:int = None
    training_size:int = None
    test_size:int = None
    test_split: float= None
    max_training_size:int= None
    max_test_size:int = None

    temporal_net_expected_shape : List[int] = None
    preprocess_datapath:str = "orca_berlin"
    data_min_max: List[float] = field(default_factory=lambda:[0.,1.])

    def __post_init__(self):
        self.dimensions, self.temporal_net_expected_shape = expected_shape(self.max_node_num,
                                                                           self.as_image,
                                                                           self.flatten,
                                                                           self.full_adjacency)
        self.training_proportion = 1. - self.test_split
