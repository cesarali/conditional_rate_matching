import torch
from conditional_rate_matching.utils.devices import check_model_devices
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping
from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader

class CRMPipeline:
    """
    """
    def __init__(self,config,model,dataloader_0,dataloader_1):
        self.model = model
        self.config = config
        self.dataloder_0 = dataloader_0
        self.dataloder_1 = dataloader_1
        self.model = model
        self.device = check_model_devices(self.model)

    def get_x0_sample(self,sample_size,train=True):
        # select the right iterator
        if hasattr(self.dataloder_0, "train"):
            dataloder_iterator = self.dataloder_0.train() if train else self.dataloder_0.test()
        else:
            dataloder_iterator = self.dataloder_0
        x_0 = sample_from_dataloader(dataloder_iterator, sample_size)
        #assert x_0.size(0) == sample_size
        return x_0

    def __call__(self, sample_size, train=True, return_path=False, return_intermediaries=False, batch_size=128):
        """
        For Conditional Rate Matching We Move Forward in Time

        :param sample_size:
        :param train:
        :param return_path:
        :param return_intermediaries:
        :param batch_size: Maximum size of the batch to process in one go.
        :return:
        """

        if return_intermediaries:
            return_path = False

        # Get the initial sample
        x_0 = self.get_x0_sample(sample_size=sample_size, train=train).to(self.device)

        # If batch_size is not set or sample_size is within the batch limit, process normally
        if batch_size is None or sample_size <= batch_size:
            x_f, x_hist, x0_hist, ts = TauLeaping(self.config, self.model, x_0, forward=True, return_path=return_path)
        else:
            # Initialize lists to store results from each batch
            x_f_batches = []
            x_hist_batches = []
            x0_hist_batches = []

            # Process in batches
            for i in range(0, sample_size, batch_size):
                x_0_batch = x_0[i:i + batch_size]
                x_f_batch, x_hist_batch, x0_hist_batch, ts = TauLeaping(self.config, self.model, x_0_batch,
                                                                              forward=True, return_path=return_path)

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

