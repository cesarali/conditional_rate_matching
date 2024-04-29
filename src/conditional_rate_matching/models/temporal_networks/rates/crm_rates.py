import torch
from torch import nn
from torch.nn.functional import softmax
from conditional_rate_matching.data.states_dataloaders_config import StatesDataloaderConfig
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig,TemporalNetworkToRateConfig

from conditional_rate_matching.models.pipelines.thermostat.thermostat_utils import load_thermostat
from conditional_rate_matching.models.temporal_networks.temporal_networks_utils import load_temporal_network
from conditional_rate_matching.models.networks.conditional_networks_utils import get_conditional_network
from conditional_rate_matching.utils.integration import integrate_quad_tensor_vec
from conditional_rate_matching.models.temporal_networks.ema import EMA

from typing import Union,Tuple,List
from functools import reduce
from torch.distributions import Categorical

def flip_rates(conditional_model,x_0,time):
    conditional_rate = conditional_model(x_0, time)
    not_x_0 = (~x_0.bool()).long()
    flip_rate = torch.gather(conditional_rate, 2, not_x_0.unsqueeze(2)).squeeze()
    return flip_rate

class TemporalToRateLinear(nn.Module):
    """
    Assigns a linear Layer
    """
    def __init__(self, config:CRMConfig, temporal_output_total):
        nn.Module.__init__(self)
        self.vocab_size = config.data1.vocab_size
        self.dimensions = config.data1.dimensions
        self.temporal_output_total = temporal_output_total

        if isinstance(config.temporal_network_to_rate,TemporalNetworkToRateConfig):
            intermediate_to_rate = config.temporal_network_to_rate.linear_reduction
        else:
            intermediate_to_rate = config.temporal_network_to_rate
        
        if intermediate_to_rate is None:
            self.temporal_to_rate = nn.Linear(temporal_output_total,self.dimensions*self.vocab_size)
        else:

            if isinstance(intermediate_to_rate,float):
                assert intermediate_to_rate < 1.
                intermediate_to_rate = int(self.dimensions * self.vocab_size * intermediate_to_rate)

            self.temporal_to_rate = nn.Sequential(
                nn.Linear(temporal_output_total, intermediate_to_rate),
                nn.Linear(intermediate_to_rate, self.dimensions * self.vocab_size)
            )

    def forward(self,x):
        return self.temporal_to_rate(x)

class TemporalToRateBernoulli(nn.Module):
    """
    Takes the output of the temporal rate as bernoulli probabilities completing 
    with 1 - p
    """
    def __init__(self, config:CRMConfig, temporal_output_total):
        nn.Module.__init__(self)

    def forward(self,x):
        #here we expect len(x.shape) == 2
        x_ = torch.zeros_like(x)
        x = torch.cat([x[:,:,None],x_[:,:,None]],dim=2)
        return x

class TemporalToRateEmpty(nn.Module):
    """
    Directly Takes the Output and converts into a rate
    """
    def __init__(self,  config:CRMConfig):
        nn.Module.__init__(self)

    def forward(self,x):
        return None

def select_temporal_to_rate(config:CRMConfig, expected_temporal_output_shape):

    temporal_output_total = reduce(lambda x, y: x * y,expected_temporal_output_shape)
    temporal_network_to_rate = config.temporal_network_to_rate

    if isinstance(temporal_network_to_rate,TemporalNetworkToRateConfig):
        type_of = temporal_network_to_rate.type_of 
        if type_of == "bernoulli":
             temporal_to_rate = TemporalToRateBernoulli(config,temporal_output_total)
        elif type_of == "empty":
            temporal_to_rate = TemporalToRateEmpty(config,temporal_output_total)
        elif type_of == "linear":
            temporal_to_rate = TemporalToRateLinear(config,temporal_output_total)
        elif type_of is None:
            config.temporal_network_to_rate.linear_reduction = None
            temporal_to_rate = TemporalToRateLinear(config,temporal_output_total)
    else:
        temporal_to_rate = TemporalToRateLinear(config,temporal_output_total)

    return temporal_to_rate

class ClassificationForwardRate(EMA,nn.Module):
    
    temporal_to_rate:Union[TemporalToRateLinear,TemporalToRateBernoulli,TemporalToRateEmpty]

    def __init__(self, config:CRMConfig, device):
        EMA.__init__(self,config)
        nn.Module.__init__(self)

        self.config = config

        if isinstance(config.data1,StatesDataloaderConfig):
            config_data = config.data0
        else:
            config_data = config.data1

        self.vocab_size = config_data.vocab_size
        self.dimensions = config_data.dimensions
        self.expected_data_shape = config_data.temporal_net_expected_shape

        self.temporal_network_to_rate = config.temporal_network_to_rate

        self.define_deep_models(config,device)
        self.define_thermostat(config)
        self.to(device)
        self.init_ema()

    def define_deep_models(self,config,device):
        self.temporal_network = load_temporal_network(config,device=device)
        self.expected_temporal_output_shape = self.temporal_network.expected_output_shape
        if self.expected_temporal_output_shape != [self.dimensions,self.vocab_size]:
            self.temporal_to_rate = select_temporal_to_rate(config,self.expected_temporal_output_shape)

    def define_thermostat(self,config):
        self.thermostat = load_thermostat(config)

    def classify(self,x,times):
        """
        this function takes the shape [batch_size,dimension,vocab_size] and make all the trsformations
        to handle the temporal network

        :param x: [batch_size,dimension,vocab_size]
        :param times:
        :return:
        """
        batch_size = x.size(0)
        expected_shape_for_temporal = torch.Size([batch_size]+self.expected_data_shape)
        current_shape = x.shape
        if current_shape != expected_shape_for_temporal:
            x = x.reshape(expected_shape_for_temporal)
        change_logits = self.temporal_network(x,times)

        if self.temporal_network.expected_output_shape != [self.dimensions,self.vocab_size]:
            change_logits = change_logits.reshape(batch_size, -1)
            change_logits = self.temporal_to_rate(change_logits)
            change_logits = change_logits.reshape(batch_size,self.dimensions,self.vocab_size)
        return change_logits

    def forward(self, x, time, conditional=None):
        """
        RATE

        :param x: [batch_size,dimensions]
        :param time:
        :return:[batch_size,dimensions,vocabulary_size]
        """
        batch_size = x.size(0)
        if len(x.shape) != 2:
            x = x.reshape(batch_size,-1)
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

        beta_integral_ = self.beta_integral(right_time_size(1.), right_time_size(time))
        w_1t = torch.exp(-self.vocab_size * beta_integral_)
        A = 1.
        B = (w_1t * self.vocab_size) / (1. - w_1t)
        C = w_1t

        change_logits = self.classify(x, time)
        change_classifier = softmax(change_logits, dim=2)

        #x = x.reshape(batch_size,self.dimensions)
        where_iam_classifier = torch.gather(change_classifier, 2, x.long().unsqueeze(2))

        rates = A + B[:,None,None]*change_classifier + C[:,None,None]*where_iam_classifier
        return rates

    #====================================================================
    # CONDITIONAL AND TRANSITIONS RATES INVOLVED
    #====================================================================
    def conditional_probability(self, x, x0, t, t0):
        """

        \begin{equation}
        P(x(t) = i|x(t_0)) = \frac{1}{s} + w_{t,t_0}\left(-\frac{1}{s} + \delta_{i,x(t_0)}\right)
        \end{equation}

        \begin{equation}
        w_{t,t_0} = e^{-S \int_{t_0}^{t} \beta(r)dr}
        \end{equation}

        """
        right_shape = lambda x: x if len(x.shape) == 3 else x[:, :, None]
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)

        t = right_time_size(t).to(x0.device)
        t0 = right_time_size(t0).to(x0.device)

        S = self.vocab_size
        integral_t0 = self.beta_integral(t, t0)
        w_t0 = torch.exp(-S * integral_t0)

        x = right_shape(x)
        x0 = right_shape(x0)

        delta_x = (x == x0).float()
        probability = 1. / S + w_t0[:, None, None] * ((-1. / S) + delta_x)

        return probability

    def telegram_bridge_probability(self, x, x1, x0, t):
        """
        \begin{equation}
        P(x_t=x|x_0,x_1) = \frac{p(x_1|x_t=x) p(x_t = x|x_0)}{p(x_1|x_0)}
        \end{equation}
        """

        P_x_to_x1 = self.conditional_probability(x1, x, t=1., t0=t)
        P_x0_to_x = self.conditional_probability(x, x0, t=t, t0=0.)
        P_x0_to_x1 = self.conditional_probability(x1, x0, t=1., t0=0.)

        conditional_transition_probability = (P_x_to_x1 * P_x0_to_x) / P_x0_to_x1
        return conditional_transition_probability

    def conditional_transition_rate(self, x, x1, t):
        """
        \begin{equation}
        f_t(\*x'|\*x,\*x_1) = \frac{p(\*x_1|x_t=\*x')}{p(\*x_1|x_t=\*x)}f_t(\*x'|\*x)
        \end{equation}
        """
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x.size(0),), t).to(x.device)
        x_to_go = self.where_to_go_x(x)

        P_xp_to_x1 = self.conditional_probability(x1, x_to_go, t=1., t0=t)
        P_x_to_x1 = self.conditional_probability(x1, x, t=1., t0=t)

        forward_rate = self.thermostat(t)[:,None,None]
        rate_transition = (P_xp_to_x1 / P_x_to_x1) * forward_rate

        return rate_transition

    def sample_x(self, x_1, x_0, time):
        device = x_1.device
        x_to_go = self.where_to_go_x(x_0)
        transition_probs = self.telegram_bridge_probability(x_to_go, x_1, x_0, time)
        sampled_x = Categorical(transition_probs).sample().to(device)
        return sampled_x

    def beta_integral(self, t1, t0):
        """
        Dummy integral for constant rate
        """
        integral = integrate_quad_tensor_vec(self.thermostat, t0, t1, 100)
        return integral

    def where_to_go_x(self, x):
        x_to_go = torch.arange(0, self.vocab_size)
        x_to_go = x_to_go[None, None, :].repeat((x.size(0), self.dimensions, 1)).float()
        x_to_go = x_to_go.to(x.device)
        return x_to_go
    
    def log_cost(self,x0,x1):
        """
        Schrodinger transport cost 

        params
        ------
        x0,x1: torch.Tensor(batch_size,dimensions) 

        returns
        -------
        cost: torch.Tensor(batch_size,batch_size)
        """
        time1 = torch.ones((x0.shape[0],), device=x1.device)
        posterior_estimate = softmax(self.classify(x1,time1),dim=1)
    
        D = self.dimensions
        S = self.vocab_size

        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x0.size(0),), t).to(x0.device)
        beta_integral_ = self.beta_integral(right_time_size(1.), right_time_size(0.))
        w_10 = torch.exp(- S* beta_integral_)
        A = torch.log(1./S - w_10*(-1./S))
        B = torch.log(1./S - w_10*(-1./S + 1.)) - A

        # we have to cross each x0 with each x1 so is an "outer probability per dimension" and then sum
        x0_ = x0.unsqueeze(-1).repeat((1,1,x0.shape[0])).permute(2,1,0).long()
        posterior_estimate_ = posterior_estimate.unsqueeze(-1).repeat((1,1,1,x0.shape[0]))
        cost = torch.gather(posterior_estimate_,2,x0_.unsqueeze(2)).squeeze()

        #include constants
        cost = A*D - B*cost

        cost = cost.sum(axis=1)
        return cost
    #======================================================================
    # VARIANCE
    #======================================================================
    def compute_first_moment(self,t,x1,x0):
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x0.size(0),), t).to(x0.device)

        t = right_time_size(t).to(x0.device)
        t1 = right_time_size(1.).to(x0.device)
        t0 = right_time_size(0.).to(x0.device)

        i = x1
        j = x0

        S = self.vocab_size
        integral_t0 = self.beta_integral(t, t0)
        integral_1t = self.beta_integral(t1, t)
        integral_10 = self.beta_integral(t1, t0)

        w_t0 = torch.exp(-S * integral_t0)[:,None,None]
        w_1t = torch.exp(-S * integral_1t)[:,None,None]
        w_10 = torch.exp(-S * integral_10)[:,None,None]
        # Kronecker delta in PyTorch
        kronecker_delta_ij = (i == j).float()[:,:,None]
        i = i[:,:,None]
        j = j[:,:,None]

        # Precompute common terms to simplify the expression
        term_S1 = S + 1  # Term involving S
        part1 = w_10 * (S * kronecker_delta_ij - 1) + 1
        part2 = S * w_10 * kronecker_delta_ij - w_10 + 1

        # Calculate each term of the first moment expression
        term1 = part1 * (S + w_1t * w_t0 * term_S1 - w_1t * term_S1 - w_t0 * term_S1 + 1) / 2
        term2 = part2 * (
                    S * i * w_1t * w_t0 * kronecker_delta_ij - i * w_1t * w_t0 + i * w_1t - j * w_1t * w_t0 + j * w_t0)

        # Combine terms to compute the first moment
        first_moment = (term1 + term2) / (part1 * part2)
        return first_moment

    def compute_second_moment(self,t,x1,x0):
        right_time_size = lambda t: t if isinstance(t, torch.Tensor) else torch.full((x0.size(0),), t).to(x0.device)

        t = right_time_size(t).to(x0.device)
        t1 = right_time_size(1.).to(x0.device)
        t0 = right_time_size(0.).to(x0.device)

        i = x1
        j = x0

        S = self.vocab_size
        integral_t0 = self.beta_integral(t, t0)
        integral_1t = self.beta_integral(t1, t)
        integral_10 = self.beta_integral(t1, t0)

        w_t0 = torch.exp(-S * integral_t0)[:,None,None]
        w_1t = torch.exp(-S * integral_1t)[:,None,None]
        w_10 = torch.exp(-S * integral_10)[:,None,None]
        # Kronecker delta in PyTorch
        kronecker_delta_ij = (i == j).float()[:,:,None]
        i = i[:,:,None]
        j = j[:,:,None]

        # Precompute common terms to simplify the expression
        term_S0 = 2 * S ** 2 + 3 * S + 1  # Term involving S
        part1 = w_10 * (S * kronecker_delta_ij - 1) + 1
        part2 = S * w_10 * kronecker_delta_ij - w_10 + 1

        # Calculate each term of the second moment expression
        term1 = part1 * (term_S0 + w_1t * w_t0 * term_S0 - w_1t * term_S0 - w_t0 * term_S0 + 1) / 6
        term2 = part2 * (
                    S * i ** 2 * w_1t * w_t0 * kronecker_delta_ij - i ** 2 * w_1t * w_t0 + i ** 2 * w_1t - j ** 2 * w_1t * w_t0 + j ** 2 * w_t0)

        # Combine terms to compute the second moment
        second_moment = (term1 + term2) / (part1 * part2)
        return second_moment

    def compute_variance_torch(self,t,x1,x0):

        mean = self.compute_first_moment(t,x1,x0)
        second_moment = self.compute_second_moment(t,x1,x0)

        return second_moment - mean**2

