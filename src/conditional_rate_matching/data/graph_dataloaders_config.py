import os
from conditional_rate_matching import data_path
from dataclasses import dataclass,asdict,field
from typing import List, Union, Optional, Tuple, Dict
from conditional_rate_matching.configs.utils import expected_shape

graph_data_path = os.path.join(data_path,"raw","graph")

@dataclass
class GraphDataloaderConfig:
    name:str = "GraphDataloader"
    dataset_name: str =None
    batch_size: int=None
    data_dir:str = graph_data_path

    max_node_num: int=None
    max_feat_num: int=None

    dimensions: int = None
    vocab_size: int = 2

    full_adjacency: bool = True
    flatten: bool = True
    as_image: bool= True
    init:str = None #ones,zeros,deg

    total_data_size:int = None
    training_size:int = None
    test_size:int = None
    test_split: float=None
    max_training_size:int =None
    max_test_size:int=None

    temporal_net_expected_shape : List[int] = None
    preprocess_datapath:str = "orca_berlin"
    data_min_max: List[float] = field(default_factory=lambda:[0.,1.])

    def __post_init__(self):
        self.number_of_upper_entries = int(self.max_node_num*(self.max_node_num-1)*.5)
        self.dimensions, self.temporal_net_expected_shape =  expected_shape(self.max_node_num, self.as_image, self.flatten, self.full_adjacency)
        self.number_of_nodes = self.max_node_num
        self.training_proportion = 1. - self.test_split

@dataclass
class EgoConfig(GraphDataloaderConfig):
    dataset_name: str = "ego_small"
    batch_size: int = 20
    test_split: float = 0.2
    max_node_num: int = 18
    max_feat_num: int = 17
    total_data_size:int = 200
    init: str = "ones"

@dataclass
class CommunitySmallConfig(GraphDataloaderConfig):
    dataset_name: str = 'community_small'
    batch_size: int = 20
    test_split: float = 0.2
    max_node_num: int = 20
    max_feat_num: int = 10
    total_data_size:int = 200
    init: str = 'ones'

@dataclass
class CommunityConfig(GraphDataloaderConfig):
    dataset_name: str = 'community'
    batch_size: int = 32
    test_split: float = 0.2
    max_node_num: int = 11
    max_feat_num: int = 10
    total_data_size:int = 1000
    init: str = 'ones'

@dataclass
class GridConfig(GraphDataloaderConfig):
    dataset_name: str = 'grid'
    batch_size: int = 32
    test_split: float = 0.2
    max_node_num: int = 361
    max_feat_num: int = 5
    total_data_size:int = 200
    init: str = 'ones'