import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from conditional_rate_matching.configs.config_files import ExperimentFiles

from conditional_rate_matching.configs.config_ctdd import CTDDConfig
from conditional_rate_matching.models.temporal_networks.ema import EMA
from conditional_rate_matching.models.generative_models.ctdd import CTDD

from conditional_rate_matching.models.metrics.metrics_utils import log_metrics
from conditional_rate_matching.models.trainers.abstract_trainer import Trainer
from conditional_rate_matching.models.trainers.abstract_trainer import TrainerState

class CTDDTrainer(Trainer):
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator
    """

    config: CTDDConfig
    generative_model: CTDD
    generative_model_class = CTDD
    name_ = "continuos_time_discrete_denoising_trainer"

    def __init__(self,
                 config:CTDDConfig,
                 experiment_files:ExperimentFiles,
                 **kwargs):
        """
        :param paths_dataloader: contains a data distribution and a target distribution (also possibly data)
        :param backward_estimator:
        :param current_model: model to be trained
        :param past_model: model as obtained in the previous sinkhorn iteration
        :param kwargs:

             the paths_dataloader is a part of the
        """
        self.config = config
        self.number_of_epochs = self.config.trainer.number_of_epochs
        device_str = self.config.trainer.device
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.generative_model = CTDD(self.config, experiment_files=experiment_files, device=self.device)
        self.dataloader = self.generative_model.dataloader_0


    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer

        :return:
        """
        self.parameters_info()
        self.generative_model.start_new_experiment()
        self.writer = SummaryWriter(self.generative_model.experiment_files.tensorboard_path)
        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.backward_rate.parameters(), lr=self.config.trainer.learning_rate)

        # CHECK DATA
        #databatch = next(self.ctdd.dataloader_0.train().__iter__())
        #databatch = self.preprocess_data(databatch)
        #CHECK LOSS
        #initial_loss = self.train_step(self.ctdd.backward_rate, databatch, 0)
        #assert torch.isnan(initial_loss).any() == False
        #assert torch.isinf(initial_loss).any() == False
        if isinstance(self.generative_model.backward_rate, EMA) and self.config.trainer.do_ema:
            self.do_ema = True
        #SAVE INITIAL STUFF
        return np.inf

    def train_step(self, databatch, number_of_training_step,epoch):
        current_model = self.generative_model.backward_rate
        with torch.autograd.set_detect_anomaly(True):
            databatch = self.preprocess_data(databatch)
            if len(databatch) > 1:
                x_adj, x_features = databatch[0],databatch[1]
            else:
                x_adj = databatch[0]

            B = x_adj.shape[0]

            # Sample a random timestep for each image
            ts = torch.rand((B,), device=self.device) * (1.0 - self.config.loss.min_time) + self.config.loss.min_time

            x_t, x_tilde, qt0, rate = self.generative_model.scheduler.add_noise(x_adj, self.generative_model.process, ts, self.device, return_dict=False)
            x_logits, p0t_reg, p0t_sig, reg_x = current_model(x_adj, ts, x_tilde)

            self.optimizer.zero_grad()
            loss_ = self.generative_model.loss(x_adj, x_tilde, qt0, rate, x_logits, reg_x, p0t_sig, p0t_reg, self.device)
            loss_.backward()
            self.optimizer.step()

            if self.do_ema:
                self.generative_model.backward_rate.update_ema()

            # SUMMARIES
            self.writer.add_scalar('training loss', loss_.item(), number_of_training_step)
            return loss_

    def test_step(self, databatch, number_of_test_step,epoch):
        current_model = self.generative_model.backward_rate
        with torch.no_grad():
            databatch = self.preprocess_data(databatch)
            if len(databatch) > 1:
                x_adj, x_features = databatch[0], databatch[1]
            else:
                x_adj = databatch[0]

            B = x_adj.shape[0]

            # Sample a random timestep for each image
            ts = torch.rand((B,), device=self.device) * (1.0 - self.config.loss.min_time) + self.config.loss.min_time
            x_t, x_tilde, qt0, rate = self.generative_model.scheduler.add_noise(x_adj, self.generative_model.process, ts, self.device, return_dict=False)
            x_logits, p0t_reg, p0t_sig, reg_x = current_model(x_adj, ts, x_tilde)
            loss_ = self.generative_model.loss(x_adj, x_tilde, qt0, rate, x_logits, reg_x, p0t_sig, p0t_reg, self.device)
            self.writer.add_scalar('test loss', loss_, number_of_test_step)

        return loss_

    def preprocess_data(self, databatch):
        if len(databatch) == 2:
            return [databatch[0].float().to(self.device),
                    databatch[1].float().to(self.device)]
        else:
            x = databatch[0]
            return [x.float().to(self.device)]

    def get_model(self):
        return self.generative_model.backward_rate

if __name__=="__main__":
    from conditional_rate_matching.configs.config_ctdd import CTDDConfig
    from conditional_rate_matching.configs.experiments_configs.ctdd.testing_graphs import small_community, community

    # Files to save the experiments_configs
    experiment_files = ExperimentFiles(experiment_name="ctdd",
                                       experiment_type="graph",
                                       experiment_indentifier="community9",
                                       delete=True)
    config = small_community(number_of_epochs=50)
    #config = community(number_of_epochs=200)

    ctdd_trainer = CTDDTrainer(config,experiment_files)
    results_,all_metrics = ctdd_trainer.train()
    print(results_)
    print(all_metrics)