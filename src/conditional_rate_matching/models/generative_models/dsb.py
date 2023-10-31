import torch
from conditional_rate_matching.configs.config_dsb import Config as SBConfig
from torch.distributions import Bernoulli
from torch import nn
from conditional_rate_matching.models.temporal_networks.ema import EMA
from torchtyping import TensorType
import math
from conditional_rate_matching.utils.flips_utils import flip_and_copy_spins,bool_to_spins
from conditional_rate_matching.data.transforms import SpinsToBinaryTensor
from torch.utils.tensorboard import SummaryWriter
from torch.optim.adam import Adam


spins_to_binary_tensor = SpinsToBinaryTensor()

class SchrodingerBridgeBackwardRate(EMA,nn.Module):
    """
    Schrödinger Bridge Rates Defines Flip Only Rates after temporal network logits
    """
    def __init__(self,
                 config:SBConfig,
                 device:torch.device):
        EMA.__init__(self, config)
        nn.Module.__init__(self)

        self.config = config

        # DATA
        self.temporal_network_shape = torch.Size(config.data.temporal_net_expected_shape)
        self.dimension = config.data.D
        self.num_spin_states = config.data.S

        if self.num_spin_states != 2:
            raise Exception("Schrodinger Bridge Implemented for Spins Only")

        self.data_min_max = config.data.data_min_max
        self.device = device

        # TIME
        self.time_embed_dim = config.temp_network.time_embed_dim
        self.temp_network = load_temp_network(self.config,self.device)

        if isinstance(self.temp_network.expected_output_shape,list):
            self.expected_output_dim = math.prod(self.temp_network.expected_output_shape)

        if not isinstance(config.temp_network,TemporalMLPConfig):
            self.flip_rate_logits = nn.Linear(self.expected_output_dim,self.dimension).to(self.device)
            self.init_weights()

        self.init_ema()

    def forward(self,
                x: TensorType["batch_size", "dimension"],
                times:TensorType["batch_size"],
                )-> torch.FloatTensor:
        batch_size = x.shape[0]

        if x.shape[1:] != self.temporal_network_shape:
            data_size = list(self.temporal_network_shape)
            data_size.insert(0, batch_size)
            data_size = torch.Size(data_size)
            x = x.reshape(data_size)

        if isinstance(self.config.temp_network,TemporalHollowTransformerConfig):
            x = spins_to_binary_tensor(x).long()

        temporal_net_logits = self.temp_network(x, times)
        if not isinstance(self.config.temp_network,TemporalMLPConfig):
            flip_rate_logits = self.flip_rate_logits(temporal_net_logits.view(batch_size,-1))
        else:
            flip_rate_logits = temporal_net_logits.squeeze()
        flip_rates = softplus(flip_rate_logits)
        return flip_rates

    def flip_rate(self,
                  x: TensorType["batch_size", "dimension"],
                  times: TensorType["batch_size"]
                  ) -> TensorType["batch_size", "dimension"]:
        flip_rate_logits = self.forward(x,times)
        return flip_rate_logits

    def init_weights(self):
        nn.init.xavier_uniform_(self.flip_rate_logits.weight)

class RealFlip:
    """

    """
    def __init__(self,config=None,device=None):
        self.device = device

    def __call__(self,phi,X_spins,current_time)->TensorType["batch_size","number_of_spins"]:
        batch_size = X_spins.size(0)
        number_of_spins = X_spins.size(1)
        X_copy, X_flipped = flip_and_copy_spins(X_spins)
        copy_time = torch.repeat_interleave(current_time, X_spins.size(1))
        transition_rates_ = phi(X_flipped, copy_time)
        transition_rates = transition_rates_.reshape(batch_size, number_of_spins, number_of_spins)
        transition_rates = torch.einsum("bii->bi", transition_rates)
        return transition_rates

    def set_device(self,device):
        self.device = device

class SteinSpinEstimator:
    """
    This function calculates $\phi(x_{\d},-x_d)$ per dimension i.e.
    the flip rate in the given dimension if one flips in this dimension

    Args
        X torch.Tensor(size=(batch_size,number_of_spins), {1.,-1.})

     :returns
        estimator torch.Tensor(size=batch_size,number_of_spins))
    """
    def __init__(self,
                 config:SBConfig,
                 device:torch.device,
                 **kwargs):
        self.stein_epsilon = config.flip_estimator.stein_epsilon
        self.stein_sample_size = config.flip_estimator.stein_sample_size

        self.epsilon_distribution = Bernoulli(torch.full((config.data.D,),
                                                         self.stein_epsilon))
        self.device = device

    def set_device(self,device):
        self.device = device

    def stein_sample_per_sample_point(self,X,S,current_time=None):
        """
        For each binary vector in our data sample x,
        we need a sample of stein epsilon to perform the averages

        Args:
            X torch.Tensor(number_of_paths,number_of_spins)
            S torch.Tensor(number_of_sample,number_of_spins)
        """
        number_of_paths = X.shape[0]
        sample_size = S.shape[0]

        S_copy = S.repeat((number_of_paths, 1))
        X_copy = X.repeat_interleave(sample_size, 0)
        if current_time is not None:
            current_time = current_time.repeat_interleave(sample_size,0)
            return X_copy, S_copy, current_time
        else:
            return X_copy, S_copy

    def __call__(self,
                 phi:SchrodingerBridgeBackwardRate,
                 X:TensorType["batch_size", "dimension"],
                 current_time:TensorType["batch_size"]):
        # HERE WE MAKE SURE THAT THE DATA IS IN SPINS
        if X.dtype == torch.bool:
            X = bool_to_spins(X)
        S = self.epsilon_distribution.sample(sample_shape=(self.stein_sample_size,)).bool().to(self.device)
        S = ~S  # Manfred's losses requieres epsilon as the probability for -1.
        S = bool_to_spins(S)

        number_of_paths = X.shape[0]
        number_of_spins = X.shape[1]

        # ESTIMATOR
        X_stein_copy,S_stein_copy,current_time = self.stein_sample_per_sample_point(X,S,current_time)
        stein_estimator = phi.flip_rate(S_stein_copy * X_stein_copy, current_time)
        stein_estimator = (1. - S_stein_copy) * stein_estimator
        stein_estimator = stein_estimator.reshape(number_of_paths,
                                                  self.stein_sample_size,
                                                  number_of_spins)

        stein_estimator = stein_estimator.mean(axis=1)
        stein_estimator = (1 / (2. * self.stein_epsilon)) * stein_estimator
        return stein_estimator

class BackwardRatioSteinEstimator:
    """
    """
    def __init__(self,
                 config:SBConfig,
                 device):
        self.dimension = config.loss.dimension_to_check

        self.flip_old_time = config.loss.flip_old_time
        self.flip_current_time =  config.loss.flip_current_time

        if config.flip_estimator == "stein":
            self.flip_estimator = SteinSpinEstimator(config, device)
        elif config.flip_estimator == "gradient":
            raise NotImplemented
        elif config.flip_estimator == "real":
            self.flip_estimator = RealFlip(config,device)

        self.device = device

    def set_device(self,
                   device):
        self.device = device
        self.flip_estimator.set_device(device)

    def __call__(self,
                 current_model: SchrodingerBridgeBackwardRate,
                 past_model: SchrodingerBridgeBackwardRate,
                 X_spins: TensorType["batch_size", "dimension"],
                 current_time: TensorType["batch_size"],
                 sinkhorn_iteration=0):
        """
        :param current_model:
        :param X_spins:
        :return:
        """

        if self.flip_old_time:
            if sinkhorn_iteration % 2 == 0:
                old_time = current_time
            else:
                old_time = 1. - current_time
        else:
            old_time = current_time

        if self.flip_current_time:
            if sinkhorn_iteration % 2 == 0:
                current_time = current_time
            else:
                current_time = 1. - current_time

        phi_new_d = current_model.flip_rate(X_spins, current_time).squeeze()
        with torch.no_grad():
            phi_old_d = past_model.flip_rate(X_spins, old_time)
            phi_old_d = phi_old_d.squeeze()

        # stein estimate
        stein_estimate = self.flip_estimator(current_model, X_spins, current_time)

        # stein estimate
        loss = (phi_new_d ** 2) - (2. * stein_estimate * phi_old_d)

        if self.dimension is not None:
            loss_d = loss[:,self.dimension]
            return loss_d.mean()
        else:
            return loss.mean()


if __name__=="__main__":
    from conditional_rate_matching.configs.config_files import ExperimentFiles
    from conditional_rate_matching.configs.config_dsb import Config as SBConfig
    from conditional_rate_matching.data.dataloaders_utils import get_dataloaders

    # Files to save the experiments
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="dirichlet",
                                       experiment_indentifier="test2",
                                       delete=True)
    experiment_files.create_directories()

    # Configuration
    config = SBConfig()
    #config = NistConfig()

    dataloader_1, dataloader_0 = get_dataloaders(config)

    #=========================================================
    # Initialize
    #=========================================================
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    config = SBConfig()
    number_of_training_steps = 0

    writer = SummaryWriter(experiment_files.tensorboard_path)

    for sinkhorn_iteration in range(config.number_of_sinkhorn):

        optimizer = Adam(model.parameters(), lr=config.learning_rate)

        for epoch in range(config.number_of_epochs):
            for batch_1, batch_0 in zip(dataloader_1, dataloader_0):

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
                    model_rate = model(sampled_x, time)
                    loss = loss_fn(model_rate, conditional_rate)


                writer.add_scalar('training loss', loss.item(), number_of_training_steps)

                # optimization
                optimizer.zero_grad()
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                number_of_training_steps += 1

                if number_of_training_steps % 100 == 0:
                    print(f"loss {round(loss.item(), 2)}")