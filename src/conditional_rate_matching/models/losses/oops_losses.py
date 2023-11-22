import torch
from conditional_rate_matching.configs.config_oops import OopsConfig

class OopsEBMLoss:

    def __init__(self,config:OopsConfig,device):
        self.config = config
        self.p_control = config.loss.p_control
        self.l2 = config.loss.p_control
    def __call__(self,model,x,x_fake):
        logp_real = model(x).squeeze()
        if self.p_control > 0:
            grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                          create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
            grad_reg = (grad_ld ** 2. / 2.).mean() * self.p_control
        else:
            grad_reg = 0.0

        logp_fake = model(x_fake).squeeze()

        obj = logp_real.mean() - logp_fake.mean()
        loss = -obj + grad_reg + self.l2 * ((logp_real ** 2.).mean() + (logp_fake ** 2.).mean())
        return loss

