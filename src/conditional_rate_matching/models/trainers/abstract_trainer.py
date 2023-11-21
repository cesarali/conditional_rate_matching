import numpy as np
from typing import Union,List
from dataclasses import field,dataclass

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.generative_models.ctdd import CTDD
from conditional_rate_matching.models.metrics.crm_metrics_utils import log_metrics

@dataclass
class TrainerState:
    model: Union[CTDD,CRM]
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
        self.average_train_loss = np.asarray(self.train_loss).mean()

    def finish_epoch(self):
        self.test_loss = []
        self.train_loss = []

    def update_training_batch(self,loss):
        self.train_loss.append(loss)
        self.number_of_training_steps += 1

    def update_test_batch(self,loss):
        self.number_of_test_step += 1
        self.test_loss.append(loss)

from abc import ABC, abstractmethod
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Assuming CTDDConfig, CTDD, and other necessary classes are defined elsewhere

class Trainer(ABC):
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator
    """

    @abstractmethod
    def parameters_info(self):
        """
        Prints information about the parameters.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Initializes the training process.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def train_step(self, current_model, databatch, number_of_training_step):
        """
        Defines a single training step.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def test_step(self, current_model, databatch, number_of_test_step):
        """
        Defines a single test step.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def preprocess_data(self, databatch):
        """
        Preprocesses the data batch.
        To be implemented by subclasses.
        """
        pass

    def train(self):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :return:
        """

        # INITIATE LOSS
        self.initialize()
        all_metrics = {}
        results_ = {}
        self.saved = False

        training_state = TrainerState(self.generative_model)
        training_state.best_loss = np.inf

        for epoch in tqdm(range(self.number_of_epochs)):
            #TRAINING
            for step, databatch in enumerate(self.generative_model.dataloader_0.train()):
                databatch = self.preprocess_data(databatch)
                # DATA
                loss = self.train_step(databatch,training_state.number_of_training_steps)
                training_state.update_training_batch(loss.item())
            training_state.set_average_train_loss()

            #VALIDATION
            for step, databatch in enumerate(self.generative_model.dataloader_0.test()):
                databatch = self.preprocess_data(databatch)
                # DATA
                loss = self.test_step(databatch,training_state.number_of_test_step)
                training_state.update_test_batch(loss.item())
            training_state.set_average_test_loss()

            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                results_ = self.save_results(training_state,epoch+1,checkpoint=True)

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if training_state.average_test_loss < training_state.best_loss:
                if self.config.trainer.warm_up_best_model_epoch < epoch or epoch == self.number_of_epochs - 1:
                    results_ = self.save_results(training_state,epoch + 1,checkpoint=False)
                training_state.best_loss = training_state.average_test_loss

            training_state.finish_epoch()

        #=====================================================
        # RESULTS FROM BEST MODEL UPDATED WITH METRICS
        #=====================================================
        experiment_dir = self.generative_model.experiment_files.experiment_dir
        if self.saved:
            self.generative_model = self.generative_model_class(experiment_dir=experiment_dir)
        all_metrics = log_metrics(self.generative_model, epoch="best", writer=self.writer)

        self.writer.close()
        return results_,all_metrics

    def save_results(self,
                     training_state:TrainerState,
                     epoch:int,
                     checkpoint:bool=True):
        RESULTS = {
            "model": training_state.model.backward_rate,
            "best_loss": training_state.best_loss,
            "training_loss":training_state.average_train_loss,
            "test_loss":training_state.average_test_loss,
        }

        if checkpoint:
            best_model_path_checkpoint = self.generative_model.experiment_files.best_model_path_checkpoint.format(epoch)
            torch.save(RESULTS,best_model_path_checkpoint)
            self.saved = True
        else:
            torch.save(RESULTS, self.generative_model.experiment_files.best_model_path)
            self.saved = True
        return RESULTS

# Rest of your class implementation...
