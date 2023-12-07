import torch
import numpy as np
from typing import Union


from conditional_rate_matching.configs.config_dsb import DSBConfig
from conditional_rate_matching.utils.devices import check_model_devices
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader

from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import ReferenceProcess,GaussianTargetRate
from conditional_rate_matching.models.temporal_networks.rates.dsb_rate import SchrodingerBridgeRate
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import paths_iterators_train,paths_iterators

class DSBPipeline:
    """
    """
    def __init__(self, config:DSBConfig, dataloader_0, dataloader_1,device=torch.device("cpu")):
        self.config = config
        self.dataloder_0 = dataloader_0
        self.dataloder_1 = dataloader_1
        self.min_t = config.pipeline.min_t
        self.device = device
        self.number_of_steps = config.pipeline.number_of_steps

    def get_time_steps(self,forward=True):
        self.timesteps = np.concatenate((np.linspace(1.0, self.min_t, self.number_of_steps), np.array([0])))
        self.timesteps = torch.Tensor(self.timesteps)
        if forward:
            self.timesteps = torch.flip(self.timesteps, dims=[0])
        return self.timesteps

    def get_initial_sample(self, sample_size, forward=True, train=True):
        if forward:
            dataloader = self.dataloder_0
        else:
            dataloader = self.dataloder_1
        # select the right iterator
        dataloder_iterator = dataloader.train() if train else dataloader.test()
        x_0 = sample_from_dataloader(dataloder_iterator, sample_size)
        return x_0

    def __call__(self,sample_size,model:Union[SchrodingerBridgeRate,ReferenceProcess],forward=True,
                 train=False,return_path=False,return_intermediaries=False,batch_size=128):
        """
        """
        if return_intermediaries:
            return_path = False

        # Get the initial sample
        x_0 = self.get_initial_sample(sample_size=sample_size,forward=forward,train=train).to(self.device)

        # If batch_size is not set or sample_size is within the batch limit, process normally
        if batch_size is None or sample_size <= batch_size:
            x_f, x_hist, x0_hist, ts = TauLeaping(self.config, model, x_0, forward=forward, return_path=return_path)
        else:
            # Initialize lists to store results from each batch
            x_f_batches = []
            x_hist_batches = []
            x0_hist_batches = []

            # Process in batches
            for i in range(0, sample_size, batch_size):
                x_0_batch = x_0[i:i + batch_size]
                x_f_batch, x_hist_batch, x0_hist_batch, ts = TauLeaping(self.config,
                                                                        model,
                                                                        x_0_batch,
                                                                        forward=forward,
                                                                        return_path=return_path)

                # Append the results from the batch
                x_f_batches.append(x_f_batch)
                if return_path or return_intermediaries:
                    x_hist_batches.append(x_hist_batch)
                    x0_hist_batches.append(x0_hist_batch)

            # Concatenate the results from all batches
            x_f = torch.cat(x_f_batches, dim=0)
            if return_path or return_intermediaries:
                x_hist = torch.cat(x_hist_batches, dim=0)
                x0_hist = torch.cat(x0_hist_batches, dim=0)

        # Return results based on flags
        if return_path or return_intermediaries:
            return x_f, x_hist, ts
        else:
            return x_f

    def sample_from_dataloader(self,dataloder_iterator, sample_size, flatten=True):
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
        if flatten:
            x_0 = x_0.reshape(sample_size, -1)
        return x_0

    def paths_iterators(self,dataloader,rate_model, forward=True, train=True):
        """
        :param config:
        :param dataloader:
        :return: x_hist,ts
        """
        device = check_model_devices(rate_model)
        # select the right iterator
        dataloder_iterator = dataloader.train() if train else dataloader.test()
        if isinstance(rate_model,ReferenceProcess):
            for databatch in dataloder_iterator:
                initial_spins = databatch[0].to(rate_model.device)
                batch_size = initial_spins.size(0)
                time = self.get_time_steps(forward).to(rate_model.device)
                spins, time = rate_model.sample_path(initial_spins, time)
                time = time[None, :].repeat(batch_size, 1)
                yield spins, time
        else:
            # return the iterator
            for databatch in dataloder_iterator:
                x_0 = databatch[0].to(device)
                x_f, x_hist, x0_hist, ts = TauLeaping(self.config, rate_model, x_0, forward=forward, return_path=True)
                batch_size = x_hist.size(0)
                ts = ts[None, :].repeat(batch_size, 1)
                yield x_hist, ts

    def paths_iterators_train(self, dataloader, rate_model, forward=True, train=True):
        """
        :return: path_chunk, time_chunk
        """
        for x_path, ts in self.paths_iterators(dataloader, rate_model, forward=forward, train=train):
            # reshape
            batch_size = x_path.size(0)
            num_time_steps = x_path.size(1)

            x_path = x_path.view(batch_size * num_time_steps, -1)
            ts = ts.view(batch_size * num_time_steps, -1)

            # divide in chunks
            perm = torch.randperm(x_path.shape[0])
            x_path = x_path[perm]
            ts = ts[perm]

            x_path = torch.chunk(x_path, num_time_steps)
            ts = torch.chunk(ts, num_time_steps)

            for path_chunk, time_chunk in zip(x_path, ts):
                yield path_chunk, time_chunk.squeeze()

    def direction_of_past_model(self,sinkhorn_iteration=0):
        if sinkhorn_iteration % 2 == 0:
            is_past_forward = True
            start_dataloader = self.dataloder_0
        else:
            is_past_forward = False
            start_dataloader = self.dataloder_1
        return start_dataloader,is_past_forward

    def sample_paths_for_training(self,
                                  past_model:Union[SchrodingerBridgeRate,ReferenceProcess],
                                  sinkhorn_iteration=0):
        start_dataloader, is_past_forward = self.direction_of_past_model(sinkhorn_iteration)

        if isinstance(past_model,ReferenceProcess):
            past_model: GaussianTargetRate
            assert sinkhorn_iteration == 0

        return self.paths_iterators_train(start_dataloader, past_model, forward=True, train=True)

    def sample_paths_for_test(self,
                                  past_model:Union[SchrodingerBridgeRate,ReferenceProcess],
                                  sinkhorn_iteration=0):
        start_dataloader, is_past_forward = self.direction_of_past_model(sinkhorn_iteration)

        if isinstance(past_model,ReferenceProcess):
            past_model: GaussianTargetRate
            assert sinkhorn_iteration == 0

        return self.paths_iterators_train(start_dataloader, past_model, forward=True, train=False)

    def histograms_paths_for_inference(self,
                                       current_model:SchrodingerBridgeRate,
                                       past_model:Union[SchrodingerBridgeRate,ReferenceProcess],
                                       sinkhorn_iteration=0,
                                       exact_backward=True):
        device = check_model_devices(current_model)
        start_dataloader, is_past_forward = self.direction_of_past_model(sinkhorn_iteration)

        past_histogram = torch.zeros(self.number_of_steps+1,self.config.data0.dimensions).to(device)
        current_histogram = torch.zeros(self.number_of_steps+1,self.config.data0.dimensions).to(device)

        # select the right iterator
        dataloder_iterator = start_dataloader.train()

        if isinstance(past_model,ReferenceProcess):
            sample_size = 0.
            for databatch in dataloder_iterator:
                initial_spins = databatch[0].to(past_model.device)
                batch_size = initial_spins.size(0)
                sample_size+=batch_size
                time = self.get_time_steps(is_past_forward).to(past_model.device)
                spins, past_time = past_model.sample_path(initial_spins, time)
                past_histogram += spins.sum(axis=0)
                if exact_backward:
                    if is_past_forward:
                        x_0 = spins[:,-1,:]
                    else:
                        x_0 = spins[:, 0, :]
                    x_f, x_hist, x0_hist, ts = TauLeaping(self.config, current_model, x_0, forward=not is_past_forward,return_path=True)
                    current_histogram += x_hist.sum(axis=0)
            past_histogram = past_histogram / sample_size
            if exact_backward:
                current_histogram = current_histogram / sample_size
        else:
            # return the iterator
            sample_size = 0.
            for databatch in dataloder_iterator:
                x_0 = databatch[0].to(device)
                x_f, x_hist, x0_hist, past_time = TauLeaping(self.config, past_model, x_0, forward=is_past_forward, return_path=True)
                past_histogram += x_hist.sum(axis=0)
                batch_size = x_0.size(0)
                sample_size+=batch_size
                if exact_backward:
                    if is_past_forward:
                        x_0 = x_hist[:,-1,:]
                    else:
                        x_0 = x_hist[:, 0, :]
                    x_f, x_hist, x0_hist, current_time = TauLeaping(self.config, current_model, x_0, forward=not is_past_forward,return_path=True)
                    current_histogram += x_hist.sum(axis=0)
            past_histogram = past_histogram/sample_size
            if exact_backward:
                current_histogram = current_histogram/sample_size

        if not exact_backward:
            start_dataloader, _ = self.direction_of_past_model(sinkhorn_iteration+1)
            sample_size = 0
            for databatch in dataloder_iterator:
                x_0 = databatch[0].to(device)
                x_f, x_hist, x0_hist, current_time = TauLeaping(self.config, current_model, x_0, forward=not is_past_forward, return_path=True)
                batch_size = x_0.size(0)
                sample_size+=batch_size
                current_histogram += x_hist.sum(axis=0)
            current_histogram = current_histogram/sample_size

        #===========================================================
        # CHOOSE HISTOGRAMS
        #===========================================================
        if is_past_forward:
            backward_histogram = current_histogram
            forward_histogram = past_histogram
            forward_time = past_time
        else:
            backward_histogram = current_histogram
            forward_histogram = past_histogram
            forward_time = current_time

        return backward_histogram,forward_histogram,forward_time