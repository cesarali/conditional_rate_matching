import numpy as np
import torch
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

from typing import List
from dataclasses import dataclass,field
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.metrics.crm_metrics_utils import log_metrics

from conditional_rate_matching.models.generative_models.crm import (
    CRM,
    sample_x,
    uniform_pair_x0_x1
)

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


def save_results(crm_state:CRMTrainerState,
                 experiment_files:ExperimentFiles,
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

if __name__=="__main__":
    from experiments.testing_MNIST import experiment_MNIST, experiment_MNIST_Convnet
    from experiments.testing_graphs import small_community

    # Files to save the experiments
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph",
                                       experiment_indentifier="metrics7",
                                       delete=True)
    # Configuration
    #config = experiment_MNIST(max_training_size=1000)
    #config = experiment_MNIST_Convnet(max_training_size=5000,max_test_size=2000)
    #config = experiment_kStates()
    config = small_community(number_of_epochs=50,berlin=True)

    #=========================================================
    # Initialize
    #=========================================================

    # all model
    crm = CRM(config=config,experiment_files=experiment_files)
    crm.start_new_experiment()

    #=========================================================
    # Training
    #=========================================================
    writer = SummaryWriter(experiment_files.tensorboard_path)
    optimizer = Adam(crm.forward_rate.parameters(), lr=config.trainer.learning_rate)
    tqdm_object = tqdm(range(config.trainer.number_of_epochs))

    state = CRMTrainerState(crm)
    for epoch in tqdm_object:
        #TRAIN LOOP
        for batch_1, batch_0 in zip(crm.dataloader_1.train(), crm.dataloader_0.train()):
            loss = train_step(config, crm.forward_rate, crm.loss_fn, batch_1, batch_0, optimizer, crm.device)

            writer.add_scalar('training loss', loss.item(), state.number_of_training_steps)
            tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            tqdm_object.refresh()  # to show immediately the update

            state.update_training_batch(loss.item())
        state.set_average_train_loss()

        # TEST LOOP
        for batch_1, batch_0 in zip(crm.dataloader_1.test(), crm.dataloader_0.test()):
            loss = test_step(config, crm.forward_rate, crm.loss_fn, batch_1, batch_0, crm.device)
            state.update_test_batch(loss.item())
        state.set_average_test_loss()

        # STORE MODELS AND EPOCHS
        if state.average_test_loss < state.best_loss:
            results = save_results(state,experiment_files,epoch,checkpoint=False)
            state.best_loss = state.average_test_loss

        if (epoch + 1) % config.trainer.save_model_epochs == 0:
            results = save_results(state, experiment_files, epoch + 1, checkpoint=True)

        if (epoch + 1) % config.trainer.save_metric_epochs == 0:
            all_metrics = log_metrics(crm=crm, epoch=epoch + 1, writer=writer)

        state.finish_epoch()
    writer.close()


