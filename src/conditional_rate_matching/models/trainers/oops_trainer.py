import os
import torch
import numpy as np
from conditional_rate_matching.configs.config_oops import OopsConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.abstract_trainer import Trainer
from conditional_rate_matching.models.generative_models.oops import Oops

class OopsTrainer(Trainer):
    """

    """

    def __init__(self,config:OopsConfig,experiment_files:ExperimentFiles):
        self.config = config

        self.sampling_steps = config.trainer.sampling_steps
        self.batch_size = config.data0.batch_size
        self.buffer_size = config.trainer.buffer_size
        self.buffer_init = config.trainer.buffer_init
        self.number_of_epochs = self.config.trainer.number_of_epochs

        device_str = self.config.trainer.device

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.generative_model = Oops(self.config, experiment_files=experiment_files, device=self.device)
        self.dataloader = self.generative_model.dataloader_0


    def preprocess_data(self, databatch):
        x = databatch[0]
        x = x.to(self.device).requires_grad_()
        return x

    def get_model(self):
        pass

    def initialize(self):


        itr = 0
        best_val_ll = -np.inf
        self.hop_dists = []
        self.all_inds = list(range(self.buffer_size))

        self.reinit_dist = torch.distributions.Bernoulli(probs=torch.tensor(self.reinit_freq))


    def choose_random_int(self,buffer):
        # choose random inds from buffer
        buffer_inds = sorted(np.random.choice(self.all_inds, self.batch_size, replace=False))
        x_buffer = buffer[buffer_inds].to(self.device)
        reinit = self.reinit_dist.sample((self.batch_size,)).to(self.device)
        x_reinit = self.init_dist.sample((self.batch_size,)).to(self.device)
        x_fake = x_reinit * reinit[:, None] + x_buffer * (1. - reinit[:, None])
        return x_fake

    def test_step(self, current_model, databatch, number_of_test_step):
        pass

    def train_step(self, current_model, databatch, number_of_training_step):

        x = self.preprocess_data(databatch)
        x_fake = self.choose_random_int(buffer)

        hops = []  # keep track of how much the sampelr moves particles around
        for k in range(args.sampling_steps):
            x_fake_new = sampler.step(x_fake.detach(), self.model).detach()
            h = (x_fake_new != x_fake).float().view(x_fake_new.size(0), -1).sum(-1).mean().item()
            hops.append(h)
            x_fake = x_fake_new
        hop_dists.append(np.mean(hops))

        # update buffer
        buffer[buffer_inds] = x_fake.detach().cpu()









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