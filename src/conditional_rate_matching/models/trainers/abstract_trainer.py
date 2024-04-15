from dataclasses import field,dataclass
from conditional_rate_matching.models.metrics.metrics_utils import log_metrics

from abc import ABC, abstractmethod
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Union,List
from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig
from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.models.generative_models.oops import Oops
from conditional_rate_matching.models.generative_models.ctdd import CTDD
from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.generative_models.dsb import DSB

from conditional_rate_matching.data.graph_dataloaders import GraphDataloaders
from conditional_rate_matching.data.image_dataloaders import NISTLoaderConfig

@dataclass
class TrainerState:
    model: Union[CTDD,CRM,DSB]
    best_loss : float = np.inf

    average_train_loss : float = 0.
    average_test_loss : float = 0.

    test_loss: List[float] = field(default_factory=lambda:[])
    train_loss: List[float] = field(default_factory=lambda:[])

    number_of_test_step:int = 0
    number_of_training_steps:int = 0

    def set_average_test_loss(self):
        if len(self.test_loss) > 0:
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

# Assuming CTDDConfig, CTDD, and other necessary classes are defined elsewhere
class Trainer(ABC):
    """
    This trainer is intended to obtain a backward process of a markov jump via
    a ratio estimator with a stein estimator
    """

    dataloader:Union[GraphDataloaders,NISTLoaderConfig] = None
    generative_model:Union[CTDD,Oops,CRM] = None
    config:Union[CTDDConfig,OopsConfig,CRMConfig] = None
    do_ema:bool = False

    def parameters_info(self):
        print("# ==================================================")
        print("# START OF TRAINING ")
        print("# ==================================================")

        print("# Current Model ************************************")

        print(self.generative_model.experiment_files.experiment_type)
        print(self.generative_model.experiment_files.experiment_name)
        print(self.generative_model.experiment_files.experiment_indentifier)

        print("# ==================================================")
        print("# Number of Epochs {0}".format(self.number_of_epochs))
        print("# ==================================================")

    @abstractmethod
    def initialize(self):
        """
        Initializes the training process.
        To be implemented by subclasses.
        """
        pass

    def initialize_(self):
        self.initialize()
        self.parameters_info()
        self.writer = SummaryWriter(self.generative_model.experiment_files.tensorboard_path)
        self.tqdm_object = tqdm(range(self.config.trainer.number_of_epochs))
        self.best_metric = np.inf


    @abstractmethod
    def train_step(self, current_model, databatch, number_of_training_step):
        """
        Defines a single training step.
        To be implemented by subclasses.
        """
        pass

    def global_training(self,training_state,all_metrics,epoch):
        return {},all_metrics

    @abstractmethod
    def test_step(self, current_model, databatch, number_of_test_step):
        """
        Defines a single test step.
        To be implemented by subclasses.
        """
        pass

    def global_test(self,training_state,all_metrics,epoch):
        return {},all_metrics

    @abstractmethod
    def preprocess_data(self, databatch):
        """
        Preprocesses the data batch.
        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_model(self):
        pass

    def train(self):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :return:
        """
        # INITIATE LOSS
        self.initialize_()
        all_metrics = {}
        results_ = {}
        self.saved = False

        training_state = TrainerState(self.generative_model)
        for epoch in self.tqdm_object:
            #TRAINING
            for step, databatch in enumerate(self.dataloader.train()):
                databatch = self.preprocess_data(databatch)
                # DATA
                loss = self.train_step(databatch,training_state.number_of_training_steps,epoch)
                loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                training_state.update_training_batch(loss_)
                self.tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                self.tqdm_object.refresh()  # to show immediately the update
                if self.config.trainer.debug:
                    break
            training_state.set_average_train_loss()
            results_,all_metrics = self.global_training(training_state,all_metrics,epoch)

            #EVALUATES VALIDATION LOSS
            if not self.config.trainer.save_model_metrics_stopping:
                for step, databatch in enumerate(self.dataloader.test()):
                    databatch = self.preprocess_data(databatch)
                    # DATA
                    loss = self.test_step(databatch,training_state.number_of_test_step,epoch)
                    loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                    training_state.update_test_batch(loss_)
                    if self.config.trainer.debug:
                        break

            # EVALUATES METRICS
            if self.config.trainer.save_model_metrics_stopping:
                if epoch > self.config.trainer.save_model_metrics_warming:
                    all_metrics = log_metrics(self.generative_model, all_metrics=all_metrics, epoch="best",writer=self.writer)

            training_state.set_average_test_loss()
            results_,all_metrics = self.global_test(training_state,all_metrics,epoch)

            # STORING MODEL CHECKPOINTS
            if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                results_ = self.save_results(training_state,epoch+1,checkpoint=True)

            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION NOT BY IMPROVED METRICS
            if not self.config.trainer.save_model_metrics_stopping:
                current_average = training_state.average_test_loss if self.config.trainer.save_model_test_stopping else training_state.average_train_loss
                if current_average < training_state.best_loss:
                    if self.config.trainer.warm_up_best_model_epoch < epoch or epoch == self.number_of_epochs - 1:
                        results_ = self.save_results(training_state,epoch + 1,checkpoint=False)
                    training_state.best_loss = training_state.average_test_loss

            #SAVE RESULTS IF IT INCREASES METRICS
            else:
                if epoch > self.config.trainer.save_model_metrics_warming:
                    if all_metrics[self.config.trainer.metric_to_save] < self.best_metric:
                        results_ = self.save_results(training_state, epoch + 1, checkpoint=False)
                        self.best_metric = all_metrics[self.config.trainer.metric_to_save]
            training_state.finish_epoch()

        #=====================================================
        # BEST MODEL IS READ AND METRICS ARE STORED
        #=====================================================
        experiment_dir = self.generative_model.experiment_files.experiment_dir
        if self.saved:
            self.generative_model = self.generative_model_class(experiment_dir=experiment_dir)
        all_metrics = log_metrics(self.generative_model, all_metrics=all_metrics, epoch="best", writer=self.writer)
        self.writer.close()

        return results_,all_metrics

    def save_results(self,
                     training_state:TrainerState,
                     epoch:int,
                     checkpoint:bool=True):
        RESULTS = {
            "model": self.get_model(),
            "best_loss": training_state.best_loss,
            "training_loss":training_state.average_train_loss,
            "test_loss":training_state.average_test_loss,
        }
        if checkpoint:
            best_model_path_checkpoint = self.generative_model.experiment_files.best_model_path_checkpoint.format(epoch)
            torch.save(RESULTS,best_model_path_checkpoint)
        else:
            torch.save(RESULTS, self.generative_model.experiment_files.best_model_path)
        self.saved = True
        return RESULTS

