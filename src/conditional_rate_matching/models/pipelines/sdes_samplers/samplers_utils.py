import torch
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping
from conditional_rate_matching.utils.devices import check_model_devices

def sample_from_dataloader_iterator(dataloder_iterator,sample_size,flatten=True):
    """
    Samples data from the dataloader until the sample_size is met.
    
    Args:
        dataloader (DataLoader): The dataloader to sample from.
        sample_size (int): The total number of samples to collect.
        flatten (bool): Whether to flatten the samples or not.
    
    Returns:
        torch.Tensor: A tensor containing the collected samples.
    """
    size_left = sample_size
    x_0 = []
    while size_left > 0:
        dataloader_iterator = iter(dataloder_iterator)
        for databatch in dataloader_iterator:
            x_ = databatch[0]
            batch_size = x_.size(0)
            take_size = min(size_left, batch_size)
            x_0.append(x_[:take_size])
            size_left -= take_size
            if size_left == 0:
                break

    x_0 = torch.vstack(x_0)
    actual_sample_size = x_0.size(0)
    if flatten:
        x_0 = x_0.reshape(actual_sample_size, -1)
    return x_0

def paths_iterators(config,dataloader,rate_model,forward=True,train=True):
    """
    :param config:
    :param dataloader:
    :return: x_hist,ts
    """
    device = check_model_devices(rate_model)
    # select the right iterator
    dataloder_iterator = dataloader.train() if train else dataloader.test()


    for databatch in dataloder_iterator:
        initial_spins = databatch[0].to(past_model.device)
        time = self.get_time_steps().to(past_model.device)
        spins, time = past_model.sample_path(initial_spins, time)

    # return the iterator
    for databatch in dataloder_iterator:
        x_0 = databatch[0].to(device)
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
            yield path_chunk, time_chunk.squeeze()

def sinkhorn_paths_iterator_train(config,dataloader_0,dataloader_1,sinkhorn_iteration=0):
    """

    :return:
    """
    return None