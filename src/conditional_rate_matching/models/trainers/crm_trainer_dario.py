
import numpy as np
import torch
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

from typing import List
from dataclasses import dataclass,field
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.metrics.crm_metrics_utils import log_metrics
from conditional_rate_matching.models.generative_models.crm import CRM, sample_x, uniform_pair_x0_x1


@dataclass
class CRMTrainerState:
    crm: CRM
    best_loss : float = np.inf

    average_train_loss : float = 0.
    average_test_loss : float = 0.

    test_loss: List[float] = field(default_factory=lambda:[])
    train_loss: List[float] = field(default_factory=lambda:[])

    number_of_test_step:int = 0
    number_of_training_steps:int = 0

    def set_average_test_loss(self):
        self.average_test_loss = np.asarray(self.test_loss).mean()

    def set_average_train_loss(self):
        self.average_test_loss = np.asarray(self.test_loss).mean()

    def finish_epoch(self):
        self.test_loss = []
        self.train_loss = []

    def update_training_batch(self,loss):
        self.train_loss.append(loss)
        self.number_of_training_steps += 1

    def update_test_batch(self,loss):
        self.number_of_test_step += 1
        self.test_loss.append(loss)


def save_results(crm_state: CRMTrainerState,
                 experiment_files: ExperimentFiles,
                 epoch: int = 0,
                 checkpoint: bool = False):
    RESULTS = {
        "model": crm_state.crm.forward_rate,
        "best_loss": crm_state.best_loss,
        "training_loss":crm_state.average_train_loss,
        "test_loss":crm_state.average_test_loss,
    }
    if checkpoint:
        torch.save(RESULTS, experiment_files.best_model_path_checkpoint.format(epoch))
    else:
        torch.save(RESULTS, experiment_files.best_model_path)

    return RESULTS

def train_step(config,model,loss_fn,batch_1,batch_0,optimizer,device):
    # data pair and time sample
    x_1, x_0 = uniform_pair_x0_x1(batch_1, batch_0)

    x_0 = x_0.float().to(device)
    x_1 = x_1.float().to(device)

    if len(x_0.shape) > 2:
        batch_size = x_0.size(0)
        x_0 = x_0.reshape(batch_size,-1)

    if len(x_1.shape) > 2:
        batch_size = x_1.size(0)
        x_1 = x_1.reshape(batch_size,-1)

    # time selection
    batch_size = x_0.size(0)
    time = torch.rand(batch_size).to(device)

    # sample x from z
    sampled_x = sample_x(config, x_1, x_0, time)

    #LOSS
    model_classification = model.classify(x_1, time)
    model_classification_ = model_classification.view(-1, config.vocab_size)
    sampled_x = sampled_x.view(-1)
    loss = loss_fn(model_classification_,sampled_x)

    # optimization
    optimizer.zero_grad()
    loss = loss.mean()
    loss.backward()
    optimizer.step()

    return loss

def test_step(config,model,loss_fn,batch_1,batch_0,device):
    with torch.no_grad():
        # data pair and time sample
        x_1, x_0 = uniform_pair_x0_x1(batch_1, batch_0)

        x_0 = x_0.float().to(device)
        x_1 = x_1.float().to(device)

        if len(x_0.shape) > 2:
            batch_size = x_0.size(0)
            x_0 = x_0.reshape(batch_size,-1)

        if len(x_1.shape) > 2:
            batch_size = x_1.size(0)
            x_1 = x_1.reshape(batch_size,-1)

        # time selection
        batch_size = x_0.size(0)
        time = torch.rand(batch_size).to(device)

        # sample x from z
        sampled_x = sample_x(config, x_1, x_0, time)

        #LOSS
        model_classification = model.classify(x_1, time)
        model_classification_ = model_classification.view(-1, config.vocab_size)
        sampled_x = sampled_x.view(-1)
        loss = loss_fn(model_classification_,sampled_x)

    return loss


class CRMTrainer:

    def __init__(self,config, experiment_files):
        self.config = config
        self.experiment_files = experiment_files
        self.crm = CRM(config=self.config, experiment_files=self.experiment_files)
        self.crm.start_new_experiment()
        self.writer = SummaryWriter(self.experiment_files.tensorboard_path)
        self.optimizer = Adam(self.crm.forward_rate.parameters(), lr=self.config.trainer.learning_rate)
        self.tqdm_object = tqdm(range(self.config.trainer.number_of_epochs))
        self.state = CRMTrainerState(self.crm)

    def train(self):

        for epoch in self.tqdm_object:
            #TRAIN LOOP
            for batch_1, batch_0 in zip(self.crm.dataloader_1.train(), self.crm.dataloader_0.train()):
                loss = train_step(self.config, self.crm.forward_rate, self.crm.loss_fn, batch_1, batch_0, self.optimizer, self.crm.device)
                self.writer.add_scalar('training loss', loss.item(), self.state.number_of_training_steps)
                self.tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                self.tqdm_object.refresh()  # to show immediately the update
                self.state.update_training_batch(loss.item())
            self.state.set_average_train_loss()

            # TEST LOOP
            for batch_1, batch_0 in zip(self.crm.dataloader_1.test(), self.crm.dataloader_0.test()):
                loss = test_step(self.config, self.crm.forward_rate, self.crm.loss_fn, batch_1, batch_0, self.crm.device)
                self.state.update_test_batch(loss.item())
            self.state.set_average_test_loss()

            # STORE MODELS AND EPOCHS
            if self.state.average_test_loss < self.state.best_loss:
                results = save_results(self.state, self.experiment_files, epoch, checkpoint=False)
                self.state.best_loss = self.state.average_test_loss

            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                results = save_results(self.state, self.experiment_files, epoch + 1, checkpoint=True)

            if (epoch + 1) % self.config.trainer.save_metric_epochs == 0:
                all_metrics = log_metrics(crm=self.crm, epoch=epoch + 1, writer=self.writer)

            self.state.finish_epoch()
        self.writer.close()

        return results, all_metrics

    