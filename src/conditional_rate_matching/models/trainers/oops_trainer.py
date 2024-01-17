import torch
import numpy as np
from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
from conditional_rate_matching.models.generative_models.oops import Oops
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.abstract_trainer import Trainer,TrainerState
from torch.optim.adam import Adam

class OopsTrainer(Trainer):
    """
    """

    generative_model_class = Oops

    def __init__(self,config:OopsConfig,experiment_files:ExperimentFiles):
        self.config = config

        self.batch_size = config.data0.batch_size
        self.number_of_epochs = self.config.trainer.number_of_epochs
        self.trainer_sampling_steps = self.config.trainer.sampler_steps_per_training_iter
        self.eval_every = int(self.config.trainer.eval_every_epochs)
        self.test_batch_size = self.config.trainer.test_batch_size
        device_str = self.config.trainer.device

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.generative_model = Oops(self.config, experiment_files=experiment_files, device=self.device)
        self.dataloader = self.generative_model.dataloader_0
        self.pipeline = self.generative_model.pipeline


    def preprocess_data(self, databatch):
        x = databatch[0]
        x = x.to(self.device).requires_grad_()
        return x

    def get_model(self):
        return self.generative_model.model


    def initialize(self):
        self.parameters_info()
        self.generative_model.start_new_experiment()
        #DEFINE OPTIMIZERS
        self.optimizer = Adam(self.generative_model.model.parameters(),
                              lr=self.config.trainer.learning_rate,
                              weight_decay=self.config.trainer.weight_decay)
        self.best_val_ll = -np.inf
        self.hop_dists = []
        return np.inf

    def test_step(self, databatch,epoch,number_of_test_step):
        return np.inf

    def train_step(self,databatch,epoch,number_of_training_step):
        current_model = self.generative_model.model
        x = databatch
        batch_size = x.size(0)
        x_fake,buffer_inds = self.pipeline.sample_fake_from_buffer(batch_size,self.device)

        # sample
        hops = []  # keep track of how much the sampler moves particles around
        for k in range(self.trainer_sampling_steps):
            x_fake_new = self.generative_model.sampler.step(x_fake.detach(), current_model).detach()
            h = (x_fake_new != x_fake).float().view(x_fake_new.size(0), -1).sum(-1).mean().item()
            hops.append(h)
            x_fake = x_fake_new
        self.hop_dists.append(np.mean(hops))

        # update buffer
        self.pipeline.buffer[buffer_inds] = x_fake.detach().cpu()

        #loss
        loss = self.generative_model.loss(current_model,x,x_fake)

        # optimization
        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('training loss', loss.item(), number_of_training_step)

        return loss

    def global_test(self,training_state:TrainerState,all_metrics,epoch):
        for databatch in self.dataloader.train():
            x,ll = self.pipeline(self.generative_model.model,self.test_batch_size,return_path=False,get_ll=True)
            all_metrics.update({"ll":ll})
            results_ = {}
            # SAVE RESULTS IF LOSS DECREASES IN VALIDATION
            if ll.item() > self.best_val_ll:
                results_ = self.save_results(training_state,epoch + 1,checkpoint=False)
                self.best_val_ll = ll.item()
            break
        return results_,all_metrics


if __name__=="__main__":
    from conditional_rate_matching.configs.configs_classes.config_oops import OopsConfig
    from conditional_rate_matching.data.image_dataloaders import NISTLoaderConfig

    # Files to save the experiments_configs
    experiment_files = ExperimentFiles(experiment_name="oops",
                                       experiment_type="mnist",
                                       experiment_indentifier="trainer_test",
                                       delete=True)
    config = OopsConfig()

    config.model_mlp.n_blocks = 1
    config.model_mlp.n_channels = 1

    config.trainer.number_of_epochs = 4
    config.trainer.sampler_steps_per_training_iter = 2
    config.trainer.test_batch_size = 10
    config.trainer.warm_up_best_model_epoch = 0
    config.trainer.debug = True
    config.trainer.save_model_epochs = 2

    config.pipeline.number_of_betas = 2
    config.pipeline.viz_every = 1
    config.data0 = NISTLoaderConfig(batch_size=2)

    oops_trainer = OopsTrainer(config,experiment_files)
    results_,all_metrics = oops_trainer.train()
    print(results_)
    print(all_metrics)




"""
while itr < args.n_iters:
    for x in train_loader:


        # update ema_model
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.data = ema_p.data * args.ema + p.data * (1. - args.ema)

        if itr % args.print_every == 0:
            my_print("({}) | ({}/iter) cur lr = {:.4f} |log p(real) = {:.4f}, "
                     "log p(fake) = {:.4f}, diff = {:.4f}, hops = {:.4f}".format(itr, st, lr, logp_real.mean().item(),
                                                                                 logp_fake.mean().item(), obj.item(),
                                                                                 hop_dists[-1]))
        if itr % args.viz_every == 0:
            plot("{}/data_{}.png".format(args.save_dir, itr), x.detach().cpu())
            plot("{}/buffer_{}.png".format(args.save_dir, itr), x_fake)

        if (itr + 1) % args.eval_every == 0:
            logZ, train_ll, val_ll, test_ll, ais_samples = ais.evaluate(ema_model, init_dist, sampler,
                                                                        train_loader, val_loader, test_loader,
                                                                        preprocess, device,
                                                                        args.eval_sampling_steps,
                                                                        args.test_batch_size)
            my_print("EMA Train log-likelihood ({}): {}".format(itr, train_ll.item()))
            my_print("EMA Valid log-likelihood ({}): {}".format(itr, val_ll.item()))
            my_print("EMA Test log-likelihood ({}): {}".format(itr, test_ll.item()))
            for _i, _x in enumerate(ais_samples):
                plot("{}/EMA_sample_{}_{}.png".format(args.save_dir, itr, _i), _x)

            model.cpu()
            d = {}
            d['model'] = model.state_dict()
            d['ema_model'] = ema_model.state_dict()
            d['buffer'] = buffer
            d['optimizer'] = optimizer.state_dict()

            if val_ll.item() > best_val_ll:
                best_val_ll = val_ll.item()
                my_print("Best valid likelihood")
                torch.save(d, "{}/best_ckpt.pt".format(args.save_dir))
            else:
                torch.save(d, "{}/ckpt.pt".format(args.save_dir))

            model.to(device)

        itr += 1
"""