import torch
import numpy as np
from torch.optim.adam import Adam
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.metrics.metrics_utils import log_metrics

from conditional_rate_matching.models.generative_models.dsb import (
    DSB
)

from conditional_rate_matching.models.trainers.abstract_trainer import Trainer,TrainerState
from conditional_rate_matching.configs.config_dsb import DSBConfig


class DSBDataloder:

    def __init__(self, data0, data1):
        self.data0 = data0
        self.data1 = data1

    def train(self):
        return zip(self.data0.train(), self.data1.train())

    def test(self):
        return zip(self.data0.test(), self.data1.test())


class DSBTrainer(Trainer):
    config: DSBConfig
    dsb: DSB
    generative_model_class = DSB

    name_ = "discrete_schrodinger_bridge_trainer"

    def __init__(self, config, experiment_files):
        self.config = config
        self.number_of_epochs = self.config.trainer.number_of_epochs

        self.starting_sinkhorn = 0
        self.number_of_sinkhorn_iterations = self.config.trainer.number_of_sinkhorn_iterations

        self.do_ema = self.config.trainer.do_ema
        device_str = self.config.trainer.device

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.dsb = DSB(self.config, experiment_files=experiment_files, device=self.device)
        self.dataloader = DSBDataloder(self.dsb.dataloader_0, self.dsb.dataloader_1)

    def preprocess_data(self, databatch):
        return databatch

    def get_model(self):
        return (self.dsb.current_rate,self.dsb.past_rate)

    def initialize(self):
        """
        Obtains initial loss to know when to save, restart the optimizer

        :return:
        """
        self.dsb.start_new_experiment()
        # DEFINE OPTIMIZERS
        self.optimizer = Adam(self.dsb.current_rate.parameters(), lr=self.config.trainer.learning_rate)
        return np.inf

    def train_step(self, current_model, past_model, databatch, number_of_training_step, sinkhorn_iteration=0):
        databatch = self.preprocess_data(databatch)
        self.dsb.pipeline()

        X_spins = databatch[0]
        current_time = databatch[1]
        # LOSS UPDATE
        loss = self.dsb.backward_ratio_estimator(current_model,
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
            databatch = self.preprocess_data(databatch)
            X_spins = databatch[0]
            current_time = databatch[1]
            loss = self.dsb.backward_ratio_estimator(current_model, past_model, X_spins, current_time)
            self.writer.add_scalar('test loss sinkhorn {0}'.format(sinkhorn_iteration), loss, number_of_test_step)
        return loss


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

        for sinkhorn_iteration in range(self.starting_sinkhorn,self.number_of_sinkhorn):
            training_state = TrainerState(self.dsb)
            training_state.best_loss = np.inf
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
                #VALIDATION
                for step, databatch in enumerate(self.dataloader.test()):
                    databatch = self.preprocess_data(databatch)
                    # DATA
                    loss = self.test_step(databatch,training_state.number_of_test_step,epoch)
                    loss_ = loss.item() if isinstance(loss, torch.Tensor) else loss
                    training_state.update_test_batch(loss_)
                    if self.config.trainer.debug:
                        break
                training_state.set_average_test_loss()
                results_,all_metrics = self.global_test(training_state,all_metrics,epoch)
                # STORING MODEL CHECKPOINTS
                if (epoch + 1) % self.config.trainer.save_model_epochs == 0:
                    results_ = self.save_results(training_state,epoch+1,checkpoint=True)
                # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
                if training_state.average_test_loss < training_state.best_loss:
                    if self.config.trainer.warm_up_best_model_epoch < epoch or epoch == self.number_of_epochs - 1:
                        results_ = self.save_results(training_state,epoch + 1,checkpoint=False)
                    training_state.best_loss = training_state.average_test_loss
                training_state.finish_epoch()
            #=====================================================
            # BEST MODEL IS READ AND METRICS ARE STORED
            #=====================================================
            experiment_dir = self.dsb.experiment_files.experiment_dir
            if self.saved:
                self.dsb = self.generative_model_class(experiment_dir=experiment_dir)
            all_metrics = log_metrics(self.dsb, all_metrics=all_metrics, epoch="best", writer=self.writer)
            self.writer.close()
        return results_,all_metrics

    def save_results(self,
                     training_state:TrainerState,
                     epoch:int,
                     checkpoint:bool=True):
        current_rate, past_rate = self.get_model()
        RESULTS = {
            "current_rate": current_rate,
            "past_rate": past_rate,
            "best_loss": training_state.best_loss,
            "training_loss":training_state.average_train_loss,
            "test_loss":training_state.average_test_loss,
        }

        if checkpoint:
            best_model_path_checkpoint = self.dsb.experiment_files.best_model_path_checkpoint.format(epoch)
            torch.save(RESULTS,best_model_path_checkpoint)
            self.saved = True
        else:
            torch.save(RESULTS, self.dsb.experiment_files.best_model_path)
            self.saved = True
        return RESULTS

if __name__ == "__main__":
    from conditional_rate_matching.configs.experiments_configs.old_experiments.testing_graphs import small_community
    from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import LogThermostatConfig
    from dataclasses import asdict
    from pprint import pprint

    # Files to save the experiments_configs
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph",
                                       experiment_indentifier="log_berlin2",
                                       delete=True)

    # Configuration
    # config = experiment_MNIST(max_training_size=1000)
    # config = experiment_MNIST_Convnet(max_training_size=5000,max_test_size=2000)
    # config = experiment_kStates()
    # config = small_community(number_of_epochs=400,berlin=True)
    config = small_community(number_of_epochs=500, berlin=True)
    # config.thermostat = LogThermostatConfig()

    pprint(asdict(config))

    crm_trainer = DSBTrainer(config, experiment_files)
    results_, all_metrics = crm_trainer.train()
    print(all_metrics)



