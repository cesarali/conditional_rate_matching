import os
from typing import List
from pprint import pprint
from dataclasses import dataclass
from dataclasses import field,asdict
from conditional_rate_matching import data_path

# model config
from conditional_rate_matching.models.temporal_networks.temporal_networks_config import TemporalMLPConfig

# data config
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig

temporal_network_configs = {
    "TemporalMLP":TemporalMLPConfig,
}

data_configs = {"NISTLoader":NISTLoaderConfig,
                "GraphDataloader":GraphDataloaderConfig,
                "StatesDataloader":StatesDataloaderConfig}

image_data_path = os.path.join(data_path,"raw")

@dataclass
class Config:

    # data
    data0: StatesDataloaderConfig = StatesDataloaderConfig()
    data1: NISTLoaderConfig = NISTLoaderConfig()

    # process
    process_name:int = "constant" # constant
    gamma:float = .9

    # temporal network
    temporal_network: TemporalMLPConfig = TemporalMLPConfig()

    # training
    number_of_epochs:int = 300
    save_model_epochs:int = 1e6
    save_metric_epochs:int = 1e6
    data_dir:str = image_data_path
    learning_rate:str = 0.01
    device:str = "cuda:0"

    metrics: List[str] = field(default_factory=lambda :["mse_histograms",
                                                        "kdmm",
                                                        "categorical_histograms"])


    #pipeline
    number_of_steps:int = 20
    num_intermediates:int = 10

    def __post_init__(self):
        if isinstance(self.data0,dict):
            self.data0 = data_configs[self.data0["name"]](**self.data0)

        if isinstance(self.data1,dict):
            self.data1 = data_configs[self.data1["name"]](**self.data1)

        if isinstance(self.temporal_network,dict):
            self.temporal_network = temporal_network_configs[self.temporal_network["name"]](**self.temporal_network)

        self.save_model_epochs = int(.5*self.number_of_epochs)
        self.save_metric_epochs = int(.5*self.number_of_epochs)

@dataclass
class NistConfig(Config):

    dataset_name_0:str = "categorical_dirichlet"
    dataset_name_1:str = "mnist"

    data_dir:str = image_data_path
    metrics: List[str] = field(default_factory=lambda:["mse_histograms",
                                                       "binary_paths_histograms",
                                                       "marginal_binary_histograms",
                                                       "mnist_plot"])


if __name__=="__main__":
    config = NistConfig()
    pprint(asdict(config))
