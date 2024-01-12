import torch
from typing import Tuple
from .ctdd_reference import ReferenceProcess
from torchtyping import TensorType

from conditional_rate_matching.configs.configs_classes.config_dsb import DSBConfig


from conditional_rate_matching.models.generative_models.spin_glass.spin_glasses_parametrized import ParametrizedSpinGlassHamiltonian
from conditional_rate_matching.models.generative_models.spin_glass.spin_glasses_configs import SpinGlassVariablesConfig
from conditional_rate_matching.models.pipelines.reference_process.reference_process_config import GlauberDynamicsConfig
from conditional_rate_matching.data.transforms import BinaryTensorToSpinsTransform
from conditional_rate_matching.data.transforms import SpinsToBinaryTensor

spins_to_binary = SpinsToBinaryTensor()
from conditional_rate_matching.models.generative_models.spin_glass.spin_glasses_parametrized import simulate_fields_and_couplings

def create_parametrized_hamiltonian(cfg,device):
    # create parametrized hamiltonian to be used in the computation of the dynamics
    if cfg.process.fom_data_hamiltonian:
        assert isinstance(cfg.data, SpinGlassVariablesConfig)
        hamiltonian = ParametrizedSpinGlassHamiltonian(cfg.data, device)
    else:
        if cfg.process.fields is not None and cfg.process.couplings is not None:
            assert isinstance(cfg.process, GlauberDynamicsConfig)
            hamiltonian = ParametrizedSpinGlassHamiltonian(cfg.process, device)
            cfg.process.fields = hamiltonian.fields.tolist()
            cfg.process.couplings = hamiltonian.couplings.tolist()
        else:
            fields, couplings = simulate_fields_and_couplings(cfg.data.dimensions)
            cfg.process.fields = fields.tolist()
            cfg.process.couplings = couplings.tolist()
            hamiltonian = ParametrizedSpinGlassHamiltonian(cfg.process, device)
    return hamiltonian


class GlauberDynamics(ReferenceProcess):
    """
    """
    def __init__(self, cfg:DSBConfig, device,rank=None):
        self.D = cfg.data0.dimensions
        self.S = cfg.data1.vocab_size
        self.as_spins = False
        self.gamma = cfg.process.gamma
        self.beta = cfg.process.beta
        self.min_t = cfg.pipeline.min_t
        self.tau = self.min_t
        self.device = device
        self.hamiltonian = ParametrizedSpinGlassHamiltonian(cfg.process,device)


    def to(self,device):
        self.device = device
        self.hamiltonian.to(self.device)
        return self

    def flip_rate(self,states_spins,times):
        all_flip_rates = self.all_flip_rates(states_spins)
        return all_flip_rates

    def transition_rates_states(self,states_spins,tau=None):
        if tau is None:
            tau = self.tau
        S = 2
        batch_size, number_of_spins = states_spins.shape
        transition_rates_ = torch.zeros(batch_size, number_of_spins, S).to(self.device)

        all_flip_rates = self.all_flip_rates(states_spins)

        batch_index = torch.arange(0, batch_size).to(self.device)
        spin_site_index = torch.arange(0, number_of_spins).to(self.device)

        states_index = spins_to_binary(states_spins).long()  # where each site is
        states_flip_index = (~states_index.bool()).long()  # where each site is going

        repeated_batch_index = torch.repeat_interleave(batch_index, number_of_spins)
        repeated_spin_site_index = spin_site_index.repeat(batch_size)
        flatten_states_index = states_index.flatten()
        flatten_states_flip_index = states_flip_index.flatten()
        flatten_all_flip_rates = all_flip_rates.flatten()

        transition_rates_[
            repeated_batch_index, repeated_spin_site_index, flatten_states_index] = 1. / tau - flatten_all_flip_rates
        transition_rates_[
            repeated_batch_index, repeated_spin_site_index, flatten_states_flip_index] = flatten_all_flip_rates
        return transition_rates_

    def selected_flip_rates(self, states, i_random):
        if not self.as_spins:
            spins_states = BinaryTensorToSpinsTransform(states)
        else:
            spins_states = states

        H_i = self.hamiltonian.selected_hamiltonian_diagonal(spins_states, i_random)
        x_i = torch.diag(spins_states[:, i_random])
        flip_rates_ = (self.gamma * torch.exp(-x_i * H_i)) / 2 * torch.cosh(H_i)
        return flip_rates_

    def all_flip_rates(self, states):
        if not self.as_spins:
            spins_states = BinaryTensorToSpinsTransform(states)
        else:
            spins_states = states
        H_i = self.hamiltonian.all_hamiltonian_diagonal(spins_states)
        flip_rates_ = (self.gamma * torch.exp(-spins_states * H_i)) / 2 * torch.cosh(H_i)
        return flip_rates_

    def rates_states_and_times(self,states,times):
        return self.all_flip_rates(states)

    def sample_path(self, start_spins, time_grid)->Tuple[TensorType["number_of_paths","number_of_time_steps","number_of_spins"],
                                                         TensorType["number_of_time_steps"]]:
        if len(start_spins.shape) == 2:
            paths = start_spins.unsqueeze(1)
        elif len(start_spins.shape) == 3:
            paths = start_spins
        else:
            print("Wrong Path From Initial Distribution, Dynamic not possible")
            raise Exception

        number_of_paths = paths.shape[0]
        number_of_spins = paths.shape[-1]
        rows_index = torch.arange(0, number_of_paths).to(self.device)
        time_index = 0

        for time_step in time_grid[1:]:
            tau = time_grid[time_index+1] - time_grid[time_index]
            time_index +=1

            states = paths[:, -1, :]

            # WHO TO FLIP
            i_random = torch.randint(0, number_of_spins, (number_of_paths,)).to(self.device)

            # EVALUATES HAMILTONIAN
            flip_rates_ = self.selected_flip_rates(states, i_random)
            flip_probabilities = tau*flip_rates_

            r = torch.rand((number_of_paths,)).to(self.device)
            where_to_flip = r < flip_probabilities

            new_states = torch.clone(states)
            index_to_change = (rows_index[torch.where(where_to_flip)], i_random[torch.where(where_to_flip)])
            if self.as_spins:
                new_states[index_to_change] = states[index_to_change] * -1.
            else:
                new_states[index_to_change] = (~states[index_to_change].bool()).float()

            paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)

        return paths, time_grid