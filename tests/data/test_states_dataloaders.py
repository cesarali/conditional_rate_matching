import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.distributions import Dirichlet,Categorical

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,BasicPipelineConfig
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders_crm
from conditional_rate_matching.models.trainers.crm_trainer import CRMDataloder


def plot_histograms_pairs(stationary_0,stationary_1,hist0=None,hist1=None):
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12, 3))

    stationary_0_ = stationary_0.squeeze().detach().numpy()
    stationary_1_ = stationary_1.squeeze().detach().numpy()

    if hist0 is not None:
        axs[0].bar(range(len(hist0)), hist0, color='blue',alpha=0.3,label="sim")
    axs[0].bar(range(len( stationary_0_)), stationary_0_, color='green',alpha=0.3,label="exp")
    axs[0].set_ylim(0,1)
    axs[0].legend(loc="best")

    if hist1 is not None:
        axs[1].bar(range(len(hist1)), hist1, color='red',alpha=0.3,label="sim")
    axs[1].bar(range(len( stationary_1_)), stationary_1_,color='green',alpha=0.3,label="exp")
    axs[1].set_ylim(0,1)
    axs[1].legend(loc="best")
    plt.show()

def sample_events_from_dirichlet(vocab_size,dirichlet_alpha,number_of_samples=300):
    simplex_distribution = Dirichlet(torch.full((vocab_size,),dirichlet_alpha))
    simplex_point = simplex_distribution.sample((1,))
    events_distribution = Categorical(simplex_point)
    events = events_distribution.sample((number_of_samples,))
    return simplex_point.squeeze(),events    

def categorical_histogram_dataloader(dataloader_0, dimensions, number_of_classes, train=True,maximum_test_sample_size = 2000):
    """
    Just the marginal per dimension

    :param dataloader_0:
    :param dimensions:
    :param number_of_classes:
    :param train:
    :return:
    """
    if hasattr(dataloader_0, "train"):
        if train:
            dataloader = dataloader_0.train()
        else:
            dataloader = dataloader_0.test()
    else:
        dataloader = dataloader_0

    histogram = torch.zeros(dimensions,number_of_classes)
    if dataloader is None:
        return histogram

    sample_size = 0.
    for databatch in dataloader:
        x_0 = databatch[0]
        if len(x_0.shape) > 2:
            batch_size = x_0.size(0)
            x_0 = x_0.reshape(batch_size,-1)
        sample_size += x_0.size(0)
        histogram += F.one_hot(x_0.long(),num_classes=number_of_classes).sum(axis=0)
        if sample_size > maximum_test_sample_size:
            break
    histogram = histogram / sample_size
    return histogram

def get_conditional_histograms_paths(crm,t_path,num_timesteps_to_plot=10):
    vocab_size = crm.config.data1.vocab_size
    dimensions = crm.config.data1.dimensions
    number_of_steps = t_path.shape[0]
    # Generate indices for the timesteps to plot
    num_timesteps_to_plot = 10

    if num_timesteps_to_plot >= number_of_steps:
        indices = range(number_of_steps)
    else:
        indices = np.linspace(0, number_of_steps - 1, num=num_timesteps_to_plot, dtype=int)
        
    times_to_plot = t_path[indices]
    conditional_histograms_paths = torch.zeros((num_timesteps_to_plot,dimensions,vocab_size))

    sample_size = 0
    crm_dataloader = CRMDataloder(crm.dataloader_0,crm.dataloader_1)
    for databatch in crm_dataloader.train():
        batch_0, batch_1 = databatch    
        # data pair and time sample
        x_1, x_0 = crm.sample_pair(batch_1,batch_0,crm.device)
        batch_size = x_1.shape[0]

        for time_index, time in enumerate(times_to_plot.detach().numpy()):
            time = torch.full((batch_size,),time)
            sampled_x = crm.forward_rate.sample_x(x_1, x_0, time)
            counts_in_sample = F.one_hot(sampled_x,num_classes=vocab_size).sum(axis=0)
            conditional_histograms_paths[time_index] += counts_in_sample
        sample_size += batch_size
        
    conditional_histograms_paths = conditional_histograms_paths/sample_size
    return conditional_histograms_paths

if __name__=="__main__":
    dimensions = 1
    vocab_size = 3
    dirichlet_alpha = 0.5

    simplex_distribution_a,events_a = sample_events_from_dirichlet(vocab_size,dirichlet_alpha,number_of_samples=300)
    simplex_distribution_b,events_b = sample_events_from_dirichlet(vocab_size,dirichlet_alpha,number_of_samples=300)

    config = CRMConfig()
    config.data1 = StatesDataloaderConfig(dimensions=1,vocab_size=vocab_size,bernoulli_probability=simplex_distribution_a)
    crm_a = CRM(config=config)

    t_path = torch.linspace(0.,1.,100)

    dataloader_0,dataloader_1,parent_dataloader = get_dataloaders_crm(config)
    databatch = next(dataloader_1.train().__iter__())
    print(databatch[0].max())
    print(databatch[0].shape)

    #get_conditional_histograms_paths(crm_a,t_path,num_timesteps_to_plot=10)


