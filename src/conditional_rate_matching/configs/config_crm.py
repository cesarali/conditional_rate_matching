from dataclasses import dataclass

@dataclass
class Config:

    # data
    number_of_spins :int = 3
    number_of_states :int = 4
    sample_size :int = 200

    dirichlet_alpha_0 :float = 0.1
    dirichlet_alpha_1 :float = 100.

    bernoulli_probability_0 :float = 0.2
    bernoulli_probability_0 :float = 0.8

    # process
    gamma :float = .9

    # model

    # temporal network
    time_embed_dim :int = 9
    hidden_dim :int = 50

    # rate

    # training
    number_of_epochs = 300
    learning_rate = 0.01
    batch_size :int = 5
    device = "cuda:0"

    #pipeline
    number_of_steps:int = 20
    num_intermediates:int = None

    def __post_init__(self):
        self.num_intermediates = int(.5*self.number_of_steps)





