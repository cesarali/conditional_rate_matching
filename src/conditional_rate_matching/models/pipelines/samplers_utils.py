import torch
from conditional_rate_matching.configs.config_crm import Config
from conditional_rate_matching.configs.config_crm import NistConfig
from conditional_rate_matching.models.pipelines.samplers import TauLeaping

def sample_from_dataloader(dataloder_iterator,sample_size,flatten=True):
    size_left = sample_size
    x_0 = []
    for databath in dataloder_iterator:
        x_ = databath[0]
        batch_size = x_.size(0)
        take_size = min(size_left, batch_size)
        x_0.append(x_[:take_size])
        size_left -= take_size
        if size_left == 0:
            break
    x_0 = torch.vstack(x_0)
    sample_size = x_0.size(0)
    x_0 = x_0.reshape(sample_size,-1)
    return x_0

def paths_iterators(config,dataloader,rate_model,forward=True,train=True):
    """
    :param config:
    :param dataloader:
    :return: x_hist,ts
    """
    # select the right iterator
    if hasattr(dataloader,"train"):
        dataloder_iterator = dataloader.train() if train else dataloader.test()
    else:
        dataloder_iterator = dataloader

    # return the iterator
    for databatch in dataloder_iterator:
        x_0 = databatch[0]
        x_f,x_hist,x0_hist,ts = TauLeaping(config, rate_model, x_0, forward=forward,return_path=True)
        batch_size = x_hist.size(0)
        ts = ts[None, :].repeat(batch_size, 1)
        yield x_hist,ts

def paths_iterators_train(config,dataloader,rate_model,forward=True,train=True):
    """

    :return: path_chunk, time_chunk
    """
    for x_path,ts in paths_iterators(config,dataloader,rate_model,forward=forward,train=train):

        # reshape
        batch_size = x_path.size(0)
        num_time_steps = x_path.size(1)

        x_path = x_path.view(batch_size*num_time_steps,-1)
        ts = ts.view(batch_size*num_time_steps,-1)

        # divide in chunks
        perm = torch.randperm(x_path.shape[0])
        x_path = x_path[perm]
        ts = ts[perm]

        x_path = torch.chunk(x_path, num_time_steps)
        ts = torch.chunk(ts, num_time_steps)

        for path_chunk, time_chunk in zip(x_path, ts):
            yield path_chunk, time_chunk

def sinkhorn_paths_iterator_train(config,dataloader_0,dataloader_1,sinkhorn_iteration=0):
    """

    :return:
    """
    return None


if __name__=="__main__":
    from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
    from conditional_rate_matching.models.generative_models.crm import constant_rate

    config = NistConfig()
    config.batch_size = 64

    dataloader_0,dataloader_1 = get_dataloaders(config)
    rate_model = lambda x, t: constant_rate(config, x, t)

    x_path,ts = next(paths_iterators_train(config,dataloader_0,rate_model,forward=True,train=True).__iter__())
    print(x_path.shape)
    print(ts.shape)



