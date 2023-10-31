import os
from dataclasses import dataclass

@dataclass
class Config:

    # data
    dataset_name_0:str = "categorical_dirichlet"
    dataset_name_1:str = "categorical_dirichlet"

    number_of_spins :int = 3
    number_of_states :int = 4
    sample_size :int = 200
    test_split:float = .2

    dirichlet_alpha_0 :float = 0.1
    dirichlet_alpha_1 :float = 100.

    bernoulli_probability_0 :float = 0.2
    bernoulli_probability_0 :float = 0.8

    # process
    process_name:int = "constant" # constant
    gamma:float = .9

    # model

    # temporal network
    time_embed_dim :int = 9
    hidden_dim :int = 50

    # rate
    loss:str = "classifier" # classifier,naive
    flip_estimator: str = "stein" # stein, gradient, flip
    # rate

    # training
    number_of_sinkhorn = 10
    number_of_epochs = 300
    learning_rate = 0.01
    batch_size :int = 5
    device = "cuda:0"

    #pipeline
    number_of_steps:int = 20
    num_intermediates:int = None

    def __post_init__(self):
        self.num_intermediates = int(.5*self.number_of_steps)

from graph_bridges import data_path
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

    def __post_init__(self):
        self.num_intermediates = int(.5*self.number_of_steps)
        self.dimension = self.number_of_spins




