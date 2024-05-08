import torch
import numpy as np
from torch.optim.adam import Adam
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.temporal_networks.ema import EMA

from conditional_rate_matching.models.generative_models.crm import (
    CRM
)

from conditional_rate_matching.models.trainers.abstract_trainer import Trainer
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.data.music_dataloaders import LankhPianoRollDataloader
from conditional_rate_matching.data.music_dataloaders_config import LakhPianoRollConfig

class CRMDataloder:

    def __init__(self,data0,data1):
        self.data0 = data0
        self.data1 = data1

    def train(self):
        return zip(self.data0.train(),self.data1.train())

    def test(self):
        return zip(self.data0.test(),self.data1.test())

class CRMTrainer(Trainer):

    config: CRMConfig
    generative_model: CRM
    generative_model_class = CRM
    name_ = "conditional_rate_matching_trainer"

    def __init__(self,config,experiment_files,crm=None):
        self.config = config
        self.number_of_epochs = self.config.trainer.number_of_epochs
        device_str = self.config.trainer.device

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        if crm is None:
            self.generative_model = CRM(self.config, experiment_files=experiment_files, device=self.device)
        else:
            self.generative_model = crm
            
        if hasattr(config.data1, "conditional_model"):
            self.dataloader = self.generative_model.parent_dataloader
        else:
            self.dataloader = CRMDataloder(self.generative_model.dataloader_0,self.generative_model.dataloader_1)

    def preprocess_data(self, databatch):
        return databatch

    def get_model(self):
        return self.generative_model.forward_rate

    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer
        :return:
        """
        if isinstance(self.generative_model.forward_rate,EMA) and self.config.trainer.do_ema:
            self.do_ema = True

        self.generative_model.start_new_experiment()
        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.forward_rate.parameters(),
                              lr=self.config.trainer.learning_rate,
                              weight_decay=self.config.trainer.weight_decay)

        self.scheduler = None

        self.loss_stats = {}
        self.loss_stats_variance = {}

        self.loss_variance_times = torch.linspace(0.001,1.,20)

        if self.config.trainer.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                    gamma=self.config.trainer.lr_decay)

        self.conditional_tau_leaping = False
        self.conditional_model = False
        if hasattr(self.config.data1, "conditional_model"):
            self.conditional_model = self.config.data1.conditional_model
            self.conditional_dimension = self.config.data1.conditional_dimension
            self.generation_dimension = self.config.data1.dimensions - self.conditional_dimension

        if hasattr(self.config.data1, "conditional_model"):
            self.bridge_conditional = self.config.data1.bridge_conditional

        if self.conditional_model and not self.bridge_conditional:
            self.conditional_tau_leaping = True

        return np.inf

    def train_step(self,databatch, number_of_training_step,epoch):
        batch_0, batch_1 = databatch
        
        # data pair and time sample
        x_1, x_0 = self.generative_model.sample_pair(batch_1,batch_0,self.device)

        if len(x_0.shape) > 2:
            batch_size = x_0.size(0)
            x_0 = x_0.reshape(batch_size,-1)

        if len(x_1.shape) > 2:
            batch_size = x_1.size(0)
            x_1 = x_1.reshape(batch_size,-1)

        # conditional model
        if self.conditional_tau_leaping:
            conditioner = x_0[:, 0:self.conditional_dimension]
            noise = x_0[:, self.conditional_dimension:]

        # time selection
        batch_size = x_0.size(0)
        time = torch.rand(batch_size).to(self.device)

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
    config = experiment_MNIST(max_training_size=1000)
    # config = experiment_MNIST_Convnet(max_training_size=5000,max_test_size=2000)
    # config = experiment_kStates()
    # config = small_community(number_of_epochs=50,berlin=True)

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


