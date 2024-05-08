import time
import torch
from torch import nn
from torch.distributions import Bernoulli
from conditional_rate_matching.models.generative_models.spin_glass.spin_glasses_configs import SpinGlassVariablesConfig
from conditional_rate_matching.models.generative_models.spin_glass.spin_states_statistics import obtain_all_spin_states

def simulate_fields_and_couplings(number_of_spins):

    #fields = torch.full((number_of_spins,),0.2)
    #couplings = torch.full((number_of_couplings,),.1)

    number_of_couplings = ParametrizedSpinGlassHamiltonian.coupling_size(number_of_spins)
    fields = torch.Tensor(size=(number_of_spins,)).normal_(0.,1./number_of_spins)
    couplings = torch.Tensor(size=(number_of_couplings,)).normal_(0.,1/number_of_spins)

    return fields,couplings

class bernoulli_spins():
    """
    Spin probabilities based on bernoulli distribution
    per site
    """
    def __init__(self,p):
        self.p = p
        self.bernoulli_distribution = Bernoulli(p)

    def sample(self,sample_shape):
        sample_one_zeros = self.bernoulli_distribution.sample(sample_shape=sample_shape)
        sample_one_zeros[torch.where(sample_one_zeros == 0)] = -1.
        return sample_one_zeros


class ParametrizedSpinGlassHamiltonian(nn.Module):
    """
    Simple Hamiltonian Model for Parameter Estimation
    """
    def __init__(self, config:SpinGlassVariablesConfig, device=torch.device("cpu")):
        """
        Parameters
        ----------

        number_of_spins: int
        obtain_partition_function: bool
            only defined for number_of_spins < 10
        """
        super(ParametrizedSpinGlassHamiltonian, self).__init__()
        self.beta = config.beta
        self.number_of_spins = config.number_of_spins
        self.device = device
        self.mcmc_sample_size = config.mcmc_sample_size
        self.number_of_mcmc_steps = config.number_of_mcmc_steps

        self.lower_diagonal_indices = torch.tril_indices(self.number_of_spins, self.number_of_spins, -1).to(self.device)
        self.number_of_couplings = self.lower_diagonal_indices[0].shape[0]

        self.flip_mask = torch.ones((self.number_of_spins, self.number_of_spins)).to(self.device)
        self.flip_mask.as_strided([self.number_of_spins], [self.number_of_spins + 1]).copy_(torch.ones(self.number_of_spins) * -1.)
        self.model_identifier = str(int(time.time()))
        self.define_parameters(config)
        self.config = config

    def obtain_parameters(self):
        return self.config

    @classmethod
    def coupling_size(self,number_of_spins):
        lower_diagonal_indices = torch.tril_indices(number_of_spins, number_of_spins, -1)
        number_of_couplings = lower_diagonal_indices[0].shape[0]
        return number_of_couplings

    @classmethod
    def sample_random_model(self,number_of_spins,couplings_sigma=None):
        number_of_couplings = self.coupling_size(number_of_spins)

        if couplings_sigma is None:
            couplings_sigma = 1/float(number_of_spins)

        couplings = torch.clone(torch.Tensor(size=(
            number_of_couplings,)).normal_(0., couplings_sigma))
        fields = torch.clone(torch.Tensor(size=(
            number_of_spins,)).normal_(0., 1 / couplings_sigma))

        return fields,couplings

    def define_parameters(self, config:SpinGlassVariablesConfig):
        couplings = config.couplings
        fields = config.fields

        self.couplings_deterministic = config.couplings_deterministic
        self.couplings_sigma = config.couplings_sigma

        # INITIALIZING COUPLINGS AND FIELDS FROM ARRAYS
        if couplings is not None and fields is not None:
            assert isinstance(couplings,(list,torch.Tensor))
            assert isinstance(fields,(list,torch.Tensor))

            if isinstance(couplings,(list,)):
                couplings = torch.Tensor(couplings)
            if isinstance(fields,(list,)):
                fields = torch.Tensor(fields)

            assert self.number_of_couplings == couplings.shape[0]
            assert self.number_of_spins == fields.shape[0]
        else:
            # INITIALIZING COUPLINGS AND FIELDS FROM FLOAT
            if self.couplings_deterministic is None:
                fields,couplings = self.sample_random_model(self.number_of_spins, self.couplings_sigma)
            else:
                if isinstance(self.couplings_deterministic,(float)):
                    couplings = torch.full((self.number_of_couplings,),self.couplings_deterministic)
                    fields = torch.full((self.number_of_spins,),self.couplings_deterministic)

        #CONVERT INTO PARAMETERS
        self.fields = nn.Parameter(fields)
        self.couplings = nn.Parameter(couplings)
        self.to(self.device)

        #========================================================================
        if config.obtain_partition_function:
            self.obtain_partition_function()
        else:
            self.partition_function = None

    def obtain_partition_function(self):
        assert self.number_of_spins < 12
        all_states = obtain_all_spin_states(self.number_of_spins).to(self.device)
        with torch.no_grad():
            self.partition_function = self(all_states)
            self.partition_function = torch.exp(-self.partition_function).sum()

    def obtain_couplings_as_matrix(self,couplings=None):
        """
        converts the parameters vectors which store only the lower diagonal
        into a full symetric matrix

        :return:
        """
        if couplings is None:
            couplings = self.couplings.to(self.device)
        coupling_matrix = torch.zeros((self.number_of_spins, self.number_of_spins)).to(self.device)
        coupling_matrix[self.lower_diagonal_indices[0], self.lower_diagonal_indices[1]] = couplings
        coupling_matrix = coupling_matrix + coupling_matrix.T
        return coupling_matrix

    def log_probability(self,states,average=True):
        """
        :return:
        """
        if self.partition_function is not None:
            if average:
                return -self.beta*self(states).mean() + torch.log(self.partition_function)
            else:
                return -self.beta * self(states).std() + torch.log(self.partition_function)
        else:
            if average:
                return -self.beta * self(states).mean()
            else:
                return -self.beta * self(states).std()

    def sample(self, mcmc_sample_size=None,number_of_mcmc_steps=None)-> torch.Tensor:
        """
        Here we follow a basic metropolis hasting algorithm

        :return:
        """
        if mcmc_sample_size is None:
            mcmc_sample_size = self.mcmc_sample_size
        if number_of_mcmc_steps is None:
            number_of_mcmc_steps = self.number_of_mcmc_steps

        with torch.no_grad():
            # we start a simple bernoulli spin distribution
            p0 = torch.Tensor(self.number_of_spins * [0.5]).to(self.device)
            initial_distribution = bernoulli_spins(p0)
            states = initial_distribution.sample(sample_shape=(mcmc_sample_size,))

            paths = states.unsqueeze(1)
            rows_index = torch.arange(0, mcmc_sample_size).to(self.device)

            # METROPOLIS HASTING
            for mcmc_index in range(number_of_mcmc_steps):
                i_random = torch.randint(0, self.number_of_spins, (mcmc_sample_size,)).to(self.device)
                index_to_change = (rows_index, i_random)

                states = paths[:,-1,:]
                new_states = torch.clone(states)
                new_states[index_to_change] = states[index_to_change] * -1.

                H_0 = self(states)
                H_1 = self(new_states)
                H = self.beta * (H_0-H_1)

                flip_probability = torch.exp(H)
                r = torch.rand((mcmc_sample_size,)).to(self.device)
                where_to_flip = r < flip_probability

                new_states = torch.clone(states)
                index_to_change = (rows_index[torch.where(where_to_flip)], i_random[torch.where(where_to_flip)])
                new_states[index_to_change] = states[index_to_change] * -1.
                paths = torch.cat([paths, new_states.unsqueeze(1)], dim=1)

            return paths

    def selected_hamiltonian_diagonal(self, states, i_random):
        coupling_matrix = self.obtain_couplings_as_matrix()
        J_i = coupling_matrix[:, i_random].T
        H_i = self.fields[i_random]
        H_i =  H_i + torch.einsum('bi,bi->b', J_i, states)
        return H_i

    def all_hamiltonian_diagonal(self,states):
        coupling_matrix = self.obtain_couplings_as_matrix()
        theta_i = self.fields[None, :]
        H_i = theta_i + torch.einsum("bij,bj->bi", coupling_matrix[None, :, :], states)
        return H_i

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Ising Hamiltonian

        Parameters
        ----------
        states:torch.Tensor
            Ising states defined by 1 or -1 spin values

        Returns
        -------
        Hamiltonian
        """
        coupling_matrix = self.obtain_couplings_as_matrix()
        H_couplings = torch.einsum('bi,bij,bj->b', states, coupling_matrix[None, :, :], states)
        H_fields = torch.einsum('bi,bi->b', self.fields[None, :], states)
        Hamiltonian = - H_couplings - H_fields
        return Hamiltonian
