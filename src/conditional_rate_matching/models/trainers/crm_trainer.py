import torch
from tqdm import tqdm
from torch.optim.adam import Adam
from torch.utils.tensorboard import SummaryWriter

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

if __name__=="__main__":
    from experiments.testing_MNIST import experiment_MNIST, experiment_MNIST_Convnet
    from experiments.testing_graphs import small_community

    # Files to save the experiments
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="graph",
                                       experiment_indentifier="graph_metrics",
                                       delete=True)
    # Configuration
    #config = experiment_MNIST(max_training_size=1000)
    #config = experiment_MNIST_Convnet(max_training_size=5000,max_test_size=2000)
    #config = experiment_kStates()
    config = small_community(number_of_epochs=50)

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

    number_of_training_steps = 0
    for epoch in tqdm_object:
        for batch_1, batch_0 in zip(crm.dataloader_1.train(), crm.dataloader_0.train()):

            loss = train_step(config, crm.forward_rate, crm.loss_fn, batch_1, batch_0, optimizer, crm.device)
            number_of_training_steps += 1

            writer.add_scalar('training loss', loss.item(), number_of_training_steps)

            tqdm_object.set_description(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            tqdm_object.refresh()  # to show immediately the update

        if (epoch + 1) % config.trainer.save_model_epochs == 0:
            results = save_results(crm, experiment_files, epoch + 1, checkpoint=True)

        if (epoch + 1) % config.trainer.save_metric_epochs == 0:
            all_metrics = log_metrics(crm=crm, epoch=epoch + 1, writer=writer)

    writer.close()


