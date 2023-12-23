import torch
import numpy as np
from torch.optim.adam import Adam
from conditional_rate_matching.configs.config_files import ExperimentFiles

from conditional_rate_matching.models.generative_models.crm import (
    CRM
)

from conditional_rate_matching.models.trainers.abstract_trainer import Trainer
from conditional_rate_matching.configs.config_crm import CRMConfig

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

    def __init__(self,config,experiment_files):
        self.config = config
        self.number_of_epochs = self.config.trainer.number_of_epochs
        device_str = self.config.trainer.device

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.generative_model = CRM(self.config, experiment_files=experiment_files, device=self.device)
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
        self.generative_model.start_new_experiment()
        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.forward_rate.parameters(), lr=self.config.trainer.learning_rate)
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

        # time selection
        batch_size = x_0.size(0)
        time = torch.rand(batch_size).to(self.device)

        # sample x from z
        sampled_x = self.generative_model.forward_rate.sample_x(x_1, x_0, time)

        # loss
        model_classification = self.generative_model.forward_rate.classify(x_1, time)
        model_classification_ = model_classification.view(-1, self.config.data1.vocab_size)
        sampled_x = sampled_x.view(-1)

        loss = self.generative_model.loss(model_classification_,sampled_x)

        if self.config.trainer.loss_regularize:
            if self.config.trainer.loss_regularize_square:
                rate_regularizer = self.generative_model.forward_rate.thermostat(time)
            else:
                rate_regularizer = self.generative_model.forward_rate.thermostat(time)**2.
            rate_regularizer = rate_regularizer[:,None]
            rate_regularizer = rate_regularizer.repeat((1,self.config.data1.dimensions)).view(-1)
            loss = rate_regularizer*loss

        loss = loss.mean()

        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('training loss', loss.item(), number_of_training_step)
        return loss

    def test_step(self,databatch,number_of_test_step,epoch):
        batch_0, batch_1 = databatch
        with torch.no_grad():
            # data pair and time sample
            x_1, x_0 = self.generative_model.sample_pair(batch_1, batch_0, self.device)

            x_0 = x_0.float().to(self.device)
            x_1 = x_1.float().to(self.device)

            if len(x_0.shape) > 2:
                batch_size = x_0.size(0)
                x_0 = x_0.reshape(batch_size,-1)

            if len(x_1.shape) > 2:
                batch_size = x_1.size(0)
                x_1 = x_1.reshape(batch_size,-1)

            # time selection
            batch_size = x_0.size(0)
            time = torch.rand(batch_size).to(self.device)

            # sample x from z
            sampled_x = self.generative_model.forward_rate.sample_x(x_1, x_0, time)

            #LOSS
            model_classification = self.generative_model.forward_rate.classify(x_1, time)
            model_classification_ = model_classification.view(-1, self.config.data1.vocab_size)
            sampled_x = sampled_x.view(-1)
            loss = self.generative_model.loss(model_classification_,sampled_x)
            if self.config.trainer.loss_regularize:
                if self.config.trainer.loss_regularize_square:
                    rate_regularizer = self.generative_model.forward_rate.thermostat(time)
                else:
                    rate_regularizer = self.generative_model.forward_rate.thermostat(time)
                rate_regularizer = rate_regularizer[:, None]
                rate_regularizer = rate_regularizer.repeat((1, self.config.data1.dimensions)).view(-1)
                loss = rate_regularizer*loss
            loss = loss.mean()
            self.writer.add_scalar('test loss', loss.item(), number_of_test_step)

        return loss


if __name__=="__main__":
    from conditional_rate_matching.configs.experiments_configs.old_experiments.testing_graphs import small_community
    from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import LogThermostatConfig
    from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_nist import experiment_nist

    from dataclasses import asdict
    from pprint import pprint

    # Files to save the experiments_configs
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="mnist",
                                       experiment_indentifier="log_potsdam",
                                       delete=True)
    # Configuration
    #config = experiment_nist(number_of_epochs=10,
    #                         dataset_name="mnist",
    #                         temporal_network_name="conv0")

    config = small_community(number_of_epochs=10)

    config.trainer.debug = True
    config.trainer.max_test_size = 1000


    crm_trainer = CRMTrainer(config,experiment_files)
    results_,all_metrics = crm_trainer.train()




