import os
from dataclasses import dataclass
from typing import List
from dataclasses import field

@dataclass
class Config:

    # data
    dataset_name_0:str = "categorical_dirichlet"
    dataset_name_1:str = "categorical_dirichlet"

    number_of_spins :int =  5
    number_of_states :int = 4
    sample_size :int = 1000
    test_split:float = .2

    dirichlet_alpha_0 :float = 100.
    dirichlet_alpha_1 :float = 0.1

    bernoulli_probability_0 :float = 0.2
    bernoulli_probability_0 :float = 0.8

    # process
    process_name:int = "constant" # constant
    gamma:float = .9

    # model

    # temporal network
    time_embed_dim :int = 39
    hidden_dim :int = 200

    # rate
    loss:str = "classifier" # classifier,naive

    # rate

    # training
    number_of_epochs:int = 300
    save_model_epochs:int = 1e6
    save_metric_epochs:int = 1e6
    maximum_test_sample_size:int=2000
    metrics: List[str] = field(default_factory=lambda :["mse_histograms","kdmm","categorical_histograms"])
    learning_rate = 0.01
    batch_size :int = 5
    device = "cuda:0"

    #pipeline
    number_of_steps:int = 20
    num_intermediates:int = None

    def __post_init__(self):
        self.num_intermediates = int(.5*self.number_of_steps)
        self.save_model_epochs = int(.5*self.number_of_epochs)
        self.save_metric_epochs = int(.5*self.number_of_epochs)

from conditional_rate_matching import data_path

image_data_path = os.path.join(data_path,"raw")

@dataclass
class NistConfig(Config):

    dataset_name_0:str = "categorical_dirichlet"
    dataset_name_1:str = "mnist"

    number_of_spins:int = 784
    number_of_states:int = 2
    sample_size:int = 1000
    as_image:bool = False
    as_spins:bool = False

    pepper_threshold:float = 0.5
    data_dir:str = image_data_path
    metrics: List[str] = field(default_factory=lambda :["mse_histograms","kdmm"])

    def __post_init__(self):
        super().__post_init__()
        self.num_intermediates = int(.5*self.number_of_steps)
        self.dimension = self.number_of_spins





