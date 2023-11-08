import torch

from conditional_rate_matching.models.pipelines.samplers import TauLeaping

from conditional_rate_matching.utils.devices import check_model_devices
from conditional_rate_matching.models.pipelines.samplers_utils import sample_from_dataloader


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

    def __call__(self,sample_size,train=True,return_path=False,return_intermediaries=False):
        """
        For Conditional Rate Matching We Move Forward in Time

        :param sample_size:
        :param train:
        :return:
        """

        if return_intermediaries:
            return_path = False

        x_0 = self.get_x0_sample(sample_size=sample_size,train=train).to(self.device)
        rate_model = self.model
        x_f, x_hist, x0_hist,ts = TauLeaping(self.config, rate_model, x_0, forward=True,return_path=return_path)
        if return_path or return_intermediaries:
            return x_f,x_hist,ts
        else:
            return x_f




