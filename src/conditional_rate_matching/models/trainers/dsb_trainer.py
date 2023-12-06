import torch
import numpy as np
from typing import Union
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch.optim.adam import Adam
from conditional_rate_matching.models.metrics.metrics_utils import log_metrics

from conditional_rate_matching.models.generative_models.dsb import (
    DSB,
    DSBExperimentsFiles
)

from conditional_rate_matching.models.pipelines.reference_process.ctdd_reference import ReferenceProcess
from conditional_rate_matching.models.temporal_networks.rates.dsb_rate import SchrodingerBridgeRate
from conditional_rate_matching.models.trainers.abstract_trainer import Trainer,TrainerState
from conditional_rate_matching.configs.config_dsb import DSBConfig
from conditional_rate_matching.models.metrics.dsb_metrics_utils import log_dsb_metrics

class DSBTrainer(Trainer):
    config: DSBConfig
    generative_model: DSB
    generative_model_class = DSB
    name_ = "discrete_schrodinger_bridge_trainer"

    def __init__(self, config, experiment_files):
        self.config = config
        self.number_of_epochs = self.config.trainer.number_of_epochs

        self.starting_sinkhorn = 0
        self.number_of_sinkhorn = self.config.trainer.number_of_sinkhorn_iterations

        self.do_ema = self.config.trainer.do_ema
        device_str = self.config.trainer.device

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.generative_model = DSB(self.config, experiment_files=experiment_files, device=self.device)

    def preprocess_data(self, databatch):
        return databatch

    def get_model(self):
        return (self.generative_model.current_rate, self.generative_model.past_rate)

    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer

        :return:
        """
        self.generative_model.start_new_experiment()
        # DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.current_rate.parameters(), lr=self.config.trainer.learning_rate)
        self.writer = SummaryWriter(self.generative_model.experiment_files.tensorboard_path)
        self.tqdm_object = tqdm(range(self.config.trainer.number_of_epochs))
        self.generative_model.experiment_files.set_sinkhorn(self.starting_sinkhorn)

        return np.inf

    def train_step(self,current_model, past_model, databatch, number_of_training_step, sinkhorn_iteration=0):
        databatch = self.preprocess_data(databatch)

        X_spins = databatch[0]
        current_time = databatch[1]
        # LOSS UPDATE
        loss = self.generative_model.backward_ratio_estimator(current_model,
                                                              past_model,
                                                              X_spins,
                                                              current_time,
                                                              sinkhorn_iteration=sinkhorn_iteration)
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.trainer.clip_grad:
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=self.config.trainer.clip_max_norm)
        self.optimizer.step()
        if self.do_ema:
            current_model.update_ema()
        # SUMMARIES
        self.writer.add_scalar('training loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_training_step)
        return loss

    def test_step(self,current_model, past_model, databatch, number_of_test_step, sinkhorn_iteration=0):
        with torch.no_grad():
            X_spins = databatch[0]
            current_time = databatch[1]
            loss = self.generative_model.backward_ratio_estimator(current_model, past_model, X_spins, current_time)
            self.writer.add_scalar('test loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_test_step)
        return loss

    def train(self):
        """
        FORWARD  means sampling from p_0 (data) -> p_1 (target)

        :return:
        """
        # INITIATE LOSS
        results_ = {}
        all_metrics = {}
        self.saved = False
        self.initialize()

        past_model = self.generative_model.past_rate
        current_model = self.generative_model.current_rate

        for sinkhorn_iteration in range(self.starting_sinkhorn,self.number_of_sinkhorn):
            print("#===================================================================")
            print("Sinkhorn ITERATION {0}".format(sinkhorn_iteration))
            print("#===================================================================")
            if sinkhorn_iteration == 0:
                past_model = self.generative_model.process
            training_state = TrainerState(self.generative_model)
            training_state.best_loss = np.inf

            for epoch in self.tqdm_object:
                #TRAINING
                for databatch in self.generative_model.pipeline.sample_paths_for_training(past_model=past_model,
                                                                                          sinkhorn_iteration=sinkhorn_iteration):
                    databatch = self.preprocess_data(databatch)
                    # DATA
                    loss = self.train_step(current_model, past_model, databatch,
                                           training_state.number_of_training_steps,
                                           sinkhorn_iteration=0)
                    loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                    training_state.update_training_batch(loss_)
                    self.tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
                    self.tqdm_object.refresh()  # to show immediately the update
                    if self.config.trainer.debug:
                        break
                training_state.set_average_train_loss()

                #VALIDATION
                for databatch in self.generative_model.pipeline.sample_paths_for_test(past_model=past_model,
                                                                                      sinkhorn_iteration=sinkhorn_iteration):
                    databatch = self.preprocess_data(databatch)
                    # DATA
                    loss = self.test_step(current_model, past_model, databatch,
                                           training_state.number_of_test_step,
                                           sinkhorn_iteration=0)
                    loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                    training_state.update_test_batch(loss_)
                    if self.config.trainer.debug:
                        break
                training_state.set_average_test_loss()

                # STORING MODEL CHECKPOINTS
                if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                    results_ = self.save_results(training_state,epoch+1,checkpoint=True,sinkhorn_iteration=sinkhorn_iteration)

                # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
                if training_state.average_test_loss < training_state.best_loss:
                    if self.config.trainer.warm_up_best_model_epoch < epoch or epoch == self.number_of_epochs - 1:
                        results_ = self.save_results(training_state,epoch + 1,checkpoint=False,sinkhorn_iteration=sinkhorn_iteration)
                    training_state.best_loss = training_state.average_test_loss
                training_state.finish_epoch()

            #=====================================================
            # BEST MODEL IS READ AND METRICS ARE STORED
            #=====================================================
            experiment_dir = self.generative_model.experiment_files.experiment_dir
            if self.saved:
                self.generative_model = self.generative_model_class(experiment_dir=experiment_dir, sinkhorn_iteration=sinkhorn_iteration)
            all_metrics = log_dsb_metrics(self.generative_model, current_model=current_model,past_model=past_model,
                                          all_metrics=all_metrics, epoch="best", sinkhorn_iteration=sinkhorn_iteration,
                                          writer=self.writer)
            #====================================================================
            # END OF SINKHORN
            #====================================================================
            past_model, current_model = self.end_of_sinkhorn(current_model,past_model,sinkhorn_iteration)
        self.writer.close()
        return results_,all_metrics

    def end_of_sinkhorn(self,
                        current_model:SchrodingerBridgeRate,
                        past_model:Union[SchrodingerBridgeRate,ReferenceProcess],sinkhorn_iteration=10):
        if isinstance(past_model,ReferenceProcess):
            past_model = self.generative_model.past_rate
        past_model.load_state_dict(current_model.state_dict())

        sinkhorn_iteration += 1
        self.generative_model.experiment_files.set_sinkhorn(sinkhorn_iteration)
        self.saved = False
        return past_model,current_model

    def save_results(self,
                     training_state:TrainerState,
                     epoch:int,
                     checkpoint:bool=True,
                     sinkhorn_iteration:int=0):
        current_rate, past_rate = self.get_model()
        RESULTS = {
            "current_rate":current_rate,
            "past_rate": past_rate,
            "best_loss": training_state.best_loss,
            "training_loss":training_state.average_train_loss,
            "test_loss":training_state.average_test_loss,
            "sinkhorn_iteration":sinkhorn_iteration,
        }
        if checkpoint:
            best_model_path_checkpoint = self.generative_model.experiment_files.best_model_path_checkpoint.format(epoch)
            torch.save(RESULTS,best_model_path_checkpoint)
            self.saved = True
        else:
            torch.save(RESULTS, self.generative_model.experiment_files.best_model_path)
            self.saved = True
        return RESULTS


if __name__ == "__main__":
    from conditional_rate_matching.configs.experiments_configs.dsb.dsb_experiments_graphs import experiment_comunity_small
    from dataclasses import asdict
    from pprint import pprint

    # Files to save the experiments_configs
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="graph",
                                           experiment_indentifier="training_test3",
                                           delete=True)

    # Configuration
    config = experiment_comunity_small(number_of_epochs=500, berlin=True)


    config.trainer.warm_up_best_model_epoch = 0
    config.trainer.number_of_sinkhorn_iterations = 1
    config.trainer.debug = False
    config.trainer.metrics = ["sb_plot"]

    crm_trainer = DSBTrainer(config, experiment_files)
    results_, all_metrics = crm_trainer.train()
    #print(all_metrics)



