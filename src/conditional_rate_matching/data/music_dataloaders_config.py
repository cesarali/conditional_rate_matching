import os
from typing import List
from dataclasses import dataclass,field
from conditional_rate_matching import data_path

image_data_path = os.path.join(data_path,"raw")


@dataclass
class LakhPianoRollConfig:
    name:str = "LakhPianoRoll"
    dataset_name:str = "lakh_roll" # emnist, fashion, mnist
    conditional_model:bool = True
    bridge_conditional:bool = True

    batch_size: int= 32
    data_dir:str = image_data_path

    conditional_dimension:int = 12
    dimensions: int = 256
    vocab_size: int = 129

    flatten: bool = True
    as_image: bool = False

    max_training_size:int = None
    max_test_size:int = None

    total_data_size: int = 6973
    training_size: int = 6000
    test_size: int = 973
    test_split: float = None

    temporal_net_expected_shape : List[int] = None
    data_min_max: List[float] = field(default_factory=lambda:[0.,128.])

    def __post_init__(self):
        self.dimensions, self.temporal_net_expected_shape = self.dimensions, [self.dimensions]
        self.number_of_labels = None
        self.test_split = self.test_size/float(self.total_data_size)


    def expected_shape(self,as_image,flatten,unet=False):
        if as_image:
            if flatten:
                shape = [1,1,784]
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
                shape = [28,28]
                dimensions = 784
        return dimensions, shape

