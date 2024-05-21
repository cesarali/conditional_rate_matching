import os
import torch
import numpy as np

from torch.utils.data import TensorDataset,DataLoader
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig
from conditional_rate_matching.data.states_dataloaders import sample_categorical_from_dirichlet

def get_data(config:LakhPianoRollConfig):
    """
    """
    data_path = config.data_dir
    train_datafile = os.path.join(data_path , "pianoroll_dataset", "pianoroll_dataset", "train.npy")
    test_datafile = os.path.join(data_path, "pianoroll_dataset", "pianoroll_dataset", "test.npy")
    train_data = np.load(train_datafile)
    test_data = np.load(test_datafile)

    if config.max_training_size is not None:
        train_data = train_data[:min(config.max_training_size, len(train_data))]

    if config.max_test_size is not None:
        test_data = test_data[:min(config.max_test_size, len(test_data))]

    descramble_datafile = os.path.join(data_path, "pianoroll_dataset", "pianoroll_dataset", "descramble_key.txt")
    descramble_key = np.loadtxt(descramble_datafile)

    return torch.Tensor(train_data),torch.Tensor(test_data),descramble_key

def get_conditional_data(train_data,test_data,config:LakhPianoRollConfig):
    config.dirichlet_alpha = 100.
    config.sample_size = config.total_data_size
    config.bernoulli_probability = None

    training_noise,test_noise,probs = sample_categorical_from_dirichlet(config=config,
                                                                  return_tensor_samples=True)

    assert training_noise.shape[0] == train_data.shape[0]
    assert test_noise.shape[0] == test_data.shape[0]

    train_data_0 = torch.cat([train_data[:,:config.conditional_dimension],training_noise[:,config.conditional_dimension:]],dim=1)
    test_data_0 = torch.cat([test_data[:,:config.conditional_dimension],test_noise[:,config.conditional_dimension:]],dim=1)

    train_data_1 = train_data
    test_data_1 = test_data

    return (train_data_1,test_data_1),(train_data_0,test_data_0)


class LankhPianoRollDataloaderDataEdge:

    def __init__(self,test_dl,train_dl,descramble):
        self.test_dl = test_dl
        self.train_dl = train_dl
        self.descramble_ = descramble


    def train(self):
        return self.train_dl

    def test(self):
        return self.test_dl

    def descramble(self,sample):
        return self.descramble_(sample)



class LankhPianoRollDataloader:
    """
    """
    music_config : LakhPianoRollConfig
    name:str = "LankhPianoRollDataloader"

    def __init__(self,music_config : LakhPianoRollConfig,bridge_data=None):
        """

        :param config:
        :param device:
        """
        self.music_config = music_config
        self.number_of_spins = self.music_config.dimensions

        train_data,test_data,descramble_key = get_data(music_config)
        training_size = train_data.shape[0]
        test_size = train_data.shape[0]
        
        if self.music_config.conditional_model:
            data_1, data_0 = get_conditional_data(train_data,test_data,self.music_config)
            self.create_conditional_dataloaders(data_1,data_0)
        else:
            self.train_dataloader_,self.test_dataloader_ = self.create_dataloaders(train_data,test_data)

        self.fake_time_ = torch.rand(self.music_config.batch_size)
        self.descramble_key = descramble_key

    def descramble(self,samples):
        return self.descramble_key[samples.flatten().astype(int)].reshape(*samples.shape)

    def train(self):
        if self.music_config.conditional_model:
            return self.train_conditional()
        else:
            return self.train_dataloader_

    def test(self):
        if self.music_config.conditional_model:
            return self.test_conditional()
        else:
            return self.test_dataloader_

    def train_conditional(self):
        for databatch in self.data_train:
            yield [databatch[0]],[databatch[1]]

    def test_conditional(self):
        for databatch in self.data_test:
            yield [databatch[0]],[databatch[1]]

    def define_sample_sizes(self):
        self.training_data_size = self.music_config.training_size
        self.test_data_size = self.music_config.test_size
        self.total_data_size = self.training_data_size + self.test_data_size
        self.music_config.training_proportion = float(self.training_data_size)/self.total_data_size

    def create_dataloaders(self,train_data, test_data):
        train_ds = TensorDataset(train_data)
        test_ds = TensorDataset(test_data)

        train_dl = DataLoader(train_ds,
                              batch_size=self.music_config.batch_size,
                              shuffle=True)

        test_dl = DataLoader(test_ds,
                             batch_size=self.music_config.batch_size,
                             shuffle=True)

        return train_dl,test_dl

    def create_conditional_dataloaders(self,data_1, data_0):
        train_data_1 = data_1[0]
        train_data_0 = data_0[0]

        test_data_0 = data_0[1]
        test_data_1 = data_1[1]

        train0_ds = TensorDataset(train_data_0)
        train1_ds = TensorDataset(train_data_1)

        test0_ds = TensorDataset(test_data_0)
        test1_ds = TensorDataset(test_data_1)

        train_ds = TensorDataset(train_data_0,train_data_1)
        test_ds = TensorDataset(test_data_0,test_data_1)

        #=======================
        # INDEPENDENT
        #=======================

        self.data0_train = DataLoader(train0_ds,
                                 batch_size=self.music_config.batch_size,
                                 shuffle=False)

        self.data1_train = DataLoader(train1_ds,
                                 batch_size=self.music_config.batch_size,
                                 shuffle=False)

        self.data0_test = DataLoader(test0_ds,
                                batch_size=self.music_config.batch_size,
                                shuffle=False)

        self.data1_test = DataLoader(test1_ds,
                                batch_size=self.music_config.batch_size,
                                shuffle=False)

        self.data1 = LankhPianoRollDataloaderDataEdge(self.data1_test,self.data1_train,descramble=self.descramble)
        self.data0 = LankhPianoRollDataloaderDataEdge(self.data0_test,self.data0_train,descramble=self.descramble)

        #=======================
        # JOIN
        #=======================

        self.data_test = DataLoader(test_ds,
                                    batch_size=self.music_config.batch_size,
                                    shuffle=True)

        self.data_train = DataLoader(train_ds,
                                     batch_size=self.music_config.batch_size,
                                     shuffle=True)