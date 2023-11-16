import os
import torch
import pickle
import numpy as np
import networkx as nx
from torchtyping import TensorType
from typing import List,Dict,Tuple,Union
from torch.utils.data import TensorDataset, DataLoader
from conditional_rate_matching.utils.graph_utils import init_features, graphs_to_tensor
from conditional_rate_matching.data.graph_dataloaders_config import GraphDataloaderConfig
import torchvision.transforms as transforms


from conditional_rate_matching.data.transforms import (
    FlattenTransform,
    UnsqueezeTensorTransform,
    SqueezeTransform,
    UnFlattenTransform,
    FromUpperDiagonalTransform,
    ToUpperDiagonalIndicesTransform
)

def from_networkx_to_spins(graph_,upper_diagonal_indices,full_adjacency=False):
    adjacency_ = nx.to_numpy_array(graph_)
    if full_adjacency:
        spins = (-1.) ** (adjacency_.flatten() + 1)
    else:
        just_upper_edges = adjacency_[upper_diagonal_indices]
        spins = (-1.) ** (just_upper_edges.flatten() + 1)
    return spins

def get_transforms(config:GraphDataloaderConfig):
    """
    :param config:

    :return: transform_list,inverse_transform_list
    """
    if config.flatten:
        if config.full_adjacency:
            if config.as_image:
                transform_list = [FlattenTransform,UnsqueezeTensorTransform(1),UnsqueezeTensorTransform(1)]
                inverse_transform_list = [SqueezeTransform,UnFlattenTransform]
            else:
                transform_list = [FlattenTransform]
                inverse_transform_list = [UnFlattenTransform]
        else:
            if config.as_image:
                transform_list = [ToUpperDiagonalIndicesTransform(), UnsqueezeTensorTransform(1), UnsqueezeTensorTransform(1)]
                inverse_transform_list = [SqueezeTransform,FromUpperDiagonalTransform()]
            else:
                transform_list = [ToUpperDiagonalIndicesTransform()]
                inverse_transform_list = [FromUpperDiagonalTransform()]
    else:
        if config.full_adjacency:
            if config.as_image:
                transform_list = [UnsqueezeTensorTransform(1)]
                inverse_transform_list = [SqueezeTransform]
            else:
                transform_list = []
                inverse_transform_list = []
        else:  # no flatten no full adjacency
            raise Exception("No Flatten and No Full Adjacency incompatible for data")

    return transform_list,inverse_transform_list

class GraphDataloaders:
    """
    """
    graph_data_config : GraphDataloaderConfig
    name:str = "GraphDataloaders"

    def __init__(self,graph_data_config):
        """

        :param config:
        :param device:
        """
        self.graph_data_config = graph_data_config
        self.number_of_spins = self.graph_data_config.dimensions

        transform_list,inverse_transform_list = get_transforms(self.graph_data_config)
        self.composed_transform = transforms.Compose(transform_list)
        self.transform_to_graph = transforms.Compose(inverse_transform_list)

        train_graph_list, test_graph_list = self.read_graph_lists()

        if graph_data_config.max_training_size is not None:
            train_graph_list = [train_graph_list[i] for i in range(min(graph_data_config.max_training_size,len(train_graph_list)))]

        if graph_data_config.max_test_size is not None:
            test_graph_list = [test_graph_list[i] for i in range(min(graph_data_config.max_test_size,len(test_graph_list)))]


        self.training_data_size = len(train_graph_list)
        self.test_data_size = len(test_graph_list)
        self.total_data_size = self.training_data_size + self.test_data_size

        self.graph_data_config.training_size = self.training_data_size
        self.graph_data_config.test_size = self.test_data_size
        self.graph_data_config.total_data_size = self.total_data_size
        self.graph_data_config.training_proportion = float(self.training_data_size)/self.total_data_size

        train_adjs_tensor,train_x_tensor = self.graph_to_tensor_and_features(train_graph_list,
                                                                             self.graph_data_config.init,
                                                                             self.graph_data_config.max_node_num,
                                                                             self.graph_data_config.max_feat_num)
        test_adjs_tensor, test_x_tensor = self.graph_to_tensor_and_features(test_graph_list,
                                                                            self.graph_data_config.init,
                                                                            self.graph_data_config.max_node_num,
                                                                            self.graph_data_config.max_feat_num)


        train_adjs_tensor = self.composed_transform(train_adjs_tensor)
        self.train_dataloader_ = self.create_dataloaders(train_adjs_tensor,train_x_tensor)

        test_adjs_tensor = self.composed_transform(test_adjs_tensor)
        self.test_dataloader_ = self.create_dataloaders(test_adjs_tensor,test_x_tensor)

        self.fake_time_ = torch.rand(self.graph_data_config.batch_size)

    def train(self):
        return self.train_dataloader_

    def test(self):
        return self.test_dataloader_

    def sample(self,sample_size=10,type="train"):
        if type == "train":
            data_iterator = self.train()
        else:
            data_iterator = self.test()

        included = 0
        x_adj_list = []
        x_features_list = []
        for databatch in data_iterator:
            x_adj = databatch[0]
            x_features = databatch[1]
            x_adj_list.append(x_adj)
            x_features_list.append(x_features)

            current_batchsize = x_adj.shape[0]
            included += current_batchsize
            if included > sample_size:
                break

        if included < sample_size:
            raise Exception("Sample Size Smaller Than Expected")

        x_adj_list = torch.vstack(x_adj_list)
        x_features_list = torch.vstack(x_features_list)

        return [x_adj_list[:sample_size],x_features_list[:sample_size]]

    def create_dataloaders(self,x_tensor, adjs_tensor):
        train_ds = TensorDataset(x_tensor, adjs_tensor)
        train_dl = DataLoader(train_ds,
                              batch_size=self.graph_data_config.batch_size,
                              shuffle=True)
        return train_dl

    def sample_to_graph(self, x_sample):
        """
        undo the transformations

        :param x_sample:
        :return:
        """
        sample_size = x_sample.size(0)
        temporal_net_expected_shape = self.graph_data_config.temporal_net_expected_shape
        expected_shape = [sample_size] + temporal_net_expected_shape
        expected_shape = torch.Size(expected_shape)
        if x_sample.shape != expected_shape:
            x_sample = x_sample.reshape(expected_shape)
        adj_matrices = self.transform_to_graph(x_sample)

        # GET GRAPH FROM GENERATIVE MODEL
        graph_list = []
        number_of_graphs = adj_matrices.shape[0]
        adj_matrices = adj_matrices.detach().cpu().numpy()
        for graph_index in range(number_of_graphs):
            graph_ = nx.from_numpy_array(adj_matrices[graph_index])
            graph_list.append(graph_)

        return graph_list


    def graph_to_tensor_and_features(self,
                                     graph_list:List[nx.Graph],
                                     init:str="zeros",
                                     max_node_num:int=None,
                                     max_feat_num:int=10)->(TensorType["number_of_graphs","max_node_num","max_node_num"],
                                                            TensorType["number_of_graphs","max_feat_num"]):
        """

        :return:adjs_tensor,x_tensor
        """
        if max_node_num is None:
            max_node_num = max([g.number_of_nodes() for g in graph_list])
        adjs_tensor = graphs_to_tensor(graph_list,max_node_num)
        x_tensor = init_features(init,adjs_tensor,max_feat_num)
        return adjs_tensor,x_tensor

    def read_graph_lists(self)->Tuple[List[nx.Graph]]:
        """

        :return: train_graph_list, test_graph_list

        """
        data_dir = self.graph_data_config.data_dir
        file_name = self.graph_data_config.dataset_name
        file_path = os.path.join(data_dir, file_name)
        with open(file_path + '.pkl', 'rb') as f:
            graph_list = pickle.load(f)
        test_size = int(self.graph_data_config.test_split * len(graph_list))

        all_node_numbers = list(map(lambda x: x.number_of_nodes(), graph_list))

        max_number_of_nodes = max(all_node_numbers)
        min_number_of_nodes = min(all_node_numbers)

        train_graph_list, test_graph_list = graph_list[test_size:], graph_list[:test_size]

        return train_graph_list, test_graph_list
