import os
from typing import List
from pathlib import Path
from dataclasses import dataclass,field
from conditional_rate_matching import data_path

image_data_path = os.path.join(data_path,"raw")

@dataclass
class DiscreteCIFAR10Config:
    data: str = "Cifar10"
    dir: Path=image_data_path
    batch_size: int= 16

    C: int = 3
    H: int = 32
    W: int = 32
    S: int = 256
    D: int = None

    shape: list = None
    random_flips = True
    preprocess_datapath:str = "graphs"
    doucet:bool = True
    as_spins:bool = False

    total_data_size:int = 60000
    training_size:int = 50000
    test_size:int = 10000

    def __post_init__(self):
        self.shape = [3,32,32]
        self.temporal_net_expected_shape = self.shape
        self.D = self.C * self.H * self.W
        self.S = 256
        self.data_min_max = [0,255]


@dataclass
class NISTLoaderConfig:
    name:str = "NISTLoader"
    dataset_name:str = "mnist" # emnist, fashion, mnist
    batch_size: int= 23
    data_dir:str = image_data_path

    max_node_num: int = None
    max_feat_num: int = None

    dimensions: int = None
    vocab_size: int = 2

    pepper_threshold: float = 0.5
    flatten: bool = True
    as_image: bool = False

    max_training_size:int = 2000
    max_test_size:int=2000

    total_data_size: int = None
    training_size: int = None
    test_size: int = None
    test_split: float = None

    temporal_net_expected_shape : List[int] = None
    data_min_max: List[float] = field(default_factory=lambda:[0.,1.])

    def __post_init__(self):
        self.dimensions, self.temporal_net_expected_shape =  self.expected_shape(self.as_image,self.flatten)
        self.number_of_nodes = self.max_node_num

    def expected_shape(self,as_image,flatten):
        if as_image:
            if flatten:
                shape = [1,1,784]
                dimensions = 784
            else:
                shape = [1, 28, 28]
                dimensions = 784
        else:
            if flatten:
                shape = [784]
                dimensions = 784
            else:
                shape = [28,28]
                dimensions = 784
        return dimensions, shape

