import os
from typing import List
from pathlib import Path
from dataclasses import dataclass,field
from conditional_rate_matching import data_path

image_data_path = os.path.join(data_path,"raw")

NUMBER_OF_LABELS = {"mnist":10,"fashion":10,"emnist":27}

@dataclass
class DistortedNISTLoaderConfig:
    name: str = "DistortedNISTLoader"
    dataset_name: str = "mnist"  # emnist, fashion, mnist
    batch_size: int = 23
    data_dir: str = image_data_path

    conditional_model:bool = True
    bridge_conditional:bool = True
    conditional_dimension:int = 12

    distortion: str = 'noise'  # noise, swirl, pixelate, half_mask
    distortion_level: float = 0.4  # 0.4, 5, 0.7, None

    max_node_num: int = None
    max_feat_num: int = None

    dimensions: int = None
    vocab_size: int = 2
    unet_resize: bool = False

    pepper_threshold: float = 0.5
    flatten: bool = True
    as_image: bool = False

    max_training_size: int = None
    max_test_size: int = None

    total_data_size: int = None
    training_size: int = None
    test_size: int = None
    test_split: float = None

    temporal_net_expected_shape: List[int] = None
    data_min_max: List[float] = field(default_factory=lambda: [0., 1.])

    def __post_init__(self):
        self.dimensions, self.temporal_net_expected_shape = self.expected_shape(self.as_image, self.flatten,
                                                                                self.unet_resize)
        self.number_of_nodes = self.max_node_num
        self.number_of_labels = NUMBER_OF_LABELS[self.dataset_name]

    def expected_shape(self, as_image, flatten, unet=False):
        if as_image:
            if flatten:
                shape = [1, 1, 784]
                dimensions = 784
            else:
                if unet:
                    shape = [1, 32, 32]
                    dimensions = 1024
                else:
                    shape = [1, 28, 28]
                    dimensions = 784
        else:
            if flatten:
                shape = [784]
                dimensions = 784
            else:
                shape = [28, 28]
                dimensions = 784
        return dimensions, shape