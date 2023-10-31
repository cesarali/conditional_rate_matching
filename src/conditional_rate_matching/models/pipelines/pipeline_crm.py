import torch

from conditional_rate_matching.models.pipelines.samplers import TauLeaping

from conditional_rate_matching.utils.devices import check_model_devices

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

        size_left = sample_size
        x_0 = []
        for databath in dataloder_iterator:
            x_ = databath[0]
            batch_size = x_.size(0)
            take_size = min(size_left,batch_size)
            x_0.append(x_[:take_size])
            size_left -= take_size
            if size_left == 0:
                break

        x_0 = torch.vstack(x_0)
        assert x_0.size(0) == sample_size

        return x_0

    def __call__(self,sample_size,train=True):
        x_0 = self.get_x0_sample(sample_size=sample_size,train=train).to(self.device)
        rate_model = self.model
        x_f, x_hist, x0_hist,ts = TauLeaping(self.config, rate_model, x_0, forward=True)
        return x_f


