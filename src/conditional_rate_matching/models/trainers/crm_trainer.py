import os
import yaml
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict
from tqdm import tqdm

from conditional_rate_matching import config_path
from conditional_rate_matching.configs.config_crm import Config,NistConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.metrics.crm_metrics_utils import log_metrics

from conditional_rate_matching.models.generative_models.crm import (
    CRM,
    sample_x,
    uniform_pair_x0_x1
)

def save_results(crm:CRM,
                 experiment_files:ExperimentFiles,
                 epoch: int = 0,
                 checkpoint: bool = False):
    RESULTS = {
        "model": crm.forward_rate,
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

if __name__=="__main__":
    from conditional_rate_matching.configs.experiments.testing_state import experiment_1
    from conditional_rate_matching.configs.experiments.testing_state import experiment_2
    from conditional_rate_matching.configs.experiments.testing_graphs import small_community

    # Files to save the experiments
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph",
                                       experiment_indentifier="save_n_loads4",
                                       delete=True)
    # Configuration
    #config = experiment_1()
    #config = experiment_2()
    #config = small_community()

    yml_path = os.path.join(config_path, "crm", "small_community.yml")
    with open(yml_path, 'r') as file:
        read_data = yaml.load(file, Loader=yaml.FullLoader)
    config = Config(**read_data)

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
    optimizer = Adam(crm.forward_rate.parameters(), lr=config.learning_rate)
    tqdm_object = tqdm(range(config.number_of_epochs))

    number_of_training_steps = 0
    for epoch in tqdm_object:
        for batch_1, batch_0 in zip(crm.dataloader_1.train(), crm.dataloader_0.train()):

            loss = train_step(config, crm.forward_rate, crm.loss_fn, batch_1, batch_0, optimizer, crm.device)
            number_of_training_steps += 1

            writer.add_scalar('training loss', loss.item(), number_of_training_steps)

            tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            tqdm_object.refresh()  # to show immediately the update

        if (epoch + 1) % config.save_model_epochs == 0:
            results = save_results(crm, experiment_files, epoch + 1, checkpoint=True)

        if (epoch + 1) % config.save_metric_epochs == 0:
            all_metrics = log_metrics(crm=crm, epoch=epoch + 1, writer=writer)

    writer.close()


