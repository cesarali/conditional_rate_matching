# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from conditional_rate_matching.configs.config_ctdd import CTDDConfig
import torch.nn.functional as F
from tqdm import tqdm

from conditional_rate_matching.models.schedulers.scheduler import CTDDScheduler
from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.data.ctdd_target import CTDDTargetData

from conditional_rate_matching.models.pipelines.samplers_utils import sample_from_dataloader
from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import ReferenceProcess
from conditional_rate_matching.models.temporal_networks.rates.ctdd_rates import BackwardRate
from typing import Optional, Tuple, Union
import torch

from conditional_rate_matching.utils.devices import check_model_devices

class CTDDPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    config : CTDDConfig
    model: BackwardRate
    reference_process: ReferenceProcess
    data0: GraphDataloaders
    target1: CTDDTargetData
    scheduler: CTDDScheduler

    def __init__(self,
                 config:CTDDConfig,
                 process:ReferenceProcess,
                 data0:GraphDataloaders,
                 data1:CTDDTargetData,
                 scheduler:CTDDScheduler):

        super().__init__()
        self.register_modules(reference_process=process,
                              data0=data0,
                              data1=data1,
                              scheduler=scheduler)
        self.ctdd_config = config
        self.D = self.ctdd_config.data0.dimensions

    def get_x0_sample(self,sample_size,train=True):
        # select the right iterator
        if hasattr(self.data1, "train"):
            dataloder_iterator = self.data1.train() if train else self.data1.test()
        else:
            dataloder_iterator = self.data1
        x_0 = sample_from_dataloader(dataloder_iterator, sample_size)
        #assert x_0.size(0) == sample_size
        return x_0

    @torch.no_grad()
    def __call__(
        self,
        model: Optional[BackwardRate] = None,
        sample_size = None,
        return_dict: bool = True,
        device :torch.device = None,
        train:bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        """

        :param model:
        :param sinkhorn_iteration:
        :param return_dict:
        :param device:
        :return:
        """
        # Sample gaussian noise to begin loop
        if sample_size is None:
            sample_size = self.ctdd_config.pipeline.sample_size
        device = check_model_devices(model)
        x = self.get_x0_sample(sample_size=sample_size, train=train).to(device)

        # set step values
        self.scheduler.set_timesteps(self.ctdd_config.pipeline.number_of_steps,
                                     self.ctdd_config.pipeline.min_t)
        timesteps = self.scheduler.timesteps
        timesteps = timesteps.to(x.device)

        for idx, t in tqdm(enumerate(timesteps[0:-1])):
            h = timesteps[idx] - timesteps[idx + 1]
            # h = h.to(x.device)
            # t = t.to(x.device)
            t = t*torch.ones((sample_size,), device=x.device)
            logits = model(x, t)
            p0t = F.softmax(logits,dim=2)  # (N, D, S)
            rates_ = self.reference_process.backward_rates_from_probability(p0t, x, t, device)

            x_new = self.scheduler.step(rates_,x,t,h).prev_sample
            x = x_new

        return x