import torch
from torch import nn
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from conditional_rate_matching.configs.config_crm import Config,NistConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from conditional_rate_matching.models.metrics.crm_metrics_utils import log_metrics

from conditional_rate_matching.models.generative_models.crm import (
    CRM,
    ConditionalBackwardRate,
    ClassificationBackwardRate,
    sample_x,
    conditional_transition_rate,
    uniform_pair_x0_x1
)

def save_results(crm:CRM,
                 experiment_files:ExperimentFiles,
                 epoch: int = 0,
                 checkpoint: bool = False):
    RESULTS = {
        "model": crm.backward_rate,
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

    batch_size = x_0.size(0)
    time = torch.rand(batch_size).to(device)

    # sample x from z
    sampled_x = sample_x(config, x_1, x_0, time)

    # conditional rate
    if config.loss == "naive":
        conditional_rate = conditional_transition_rate(config, sampled_x, x_1, time)
        model_rate = model(sampled_x.float(), time)
        loss = loss_fn(model_rate, conditional_rate)
    elif config.loss == "classifier":
        model_classification = model.classify(x_1, time)
        model_classification_ = model_classification.view(-1, config.number_of_states)
        sampled_x = sampled_x.view(-1)
        loss = loss_fn(model_classification_,
                       sampled_x)

    # optimization
    optimizer.zero_grad()
    loss = loss.mean()
    loss.backward()
    optimizer.step()

    return loss

if __name__=="__main__":
    # Files to save the experiments
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="mnist",
                                       experiment_indentifier="save_n_loads7",
                                       delete=True)
    # Configuration
    #config = Config(number_of_epochs=10,number_of_states=2)
    config = NistConfig(number_of_epochs=10,
                        hidden_dim=500,
                        time_embed_dim=250,
                        batch_size=128,
                        sample_size=500,
                        number_of_steps=100,
                        maximum_test_sample_size=200,
                        num_intermediates=5)

    #=====================================================
    # DATA STUFF
    #=====================================================

    dataloader_0, dataloader_1 = get_dataloaders(config)

    #=========================================================
    # Initialize
    #=========================================================

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    config.loss = "classifier"

    if config.loss == "naive":
        model = ConditionalBackwardRate(config, device)
        loss_fn = nn.MSELoss()
    elif config.loss == "classifier":
        model = ClassificationBackwardRate(config, device).to(device)
        loss_fn = nn.CrossEntropyLoss()

    # all model
    crm = CRM(config=config,experiment_files=experiment_files)
    crm.start_new_experiment()

    #=========================================================
    # Training
    #=========================================================
    writer = SummaryWriter(experiment_files.tensorboard_path)
    optimizer = Adam(crm.backward_rate.parameters(), lr=config.learning_rate)
    tqdm_object = tqdm(range(config.number_of_epochs))

    number_of_training_steps = 0
    for epoch in tqdm_object:
        for batch_1, batch_0 in zip(dataloader_1, dataloader_0):

            loss = train_step(config,crm.backward_rate,crm.loss_fn,batch_1,batch_0,optimizer,device)
            number_of_training_steps += 1

            writer.add_scalar('training loss', loss.item(), number_of_training_steps)

            tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            tqdm_object.refresh()  # to show immediately the update

        if (epoch + 1) % config.save_model_epochs == 0:
            results = save_results(crm, experiment_files, epoch + 1, checkpoint=True)

        if (epoch + 1) % config.save_metric_epochs == 0:
            all_metrics = log_metrics(crm=crm, epoch=epoch + 1, writer=writer)

    writer.close()


