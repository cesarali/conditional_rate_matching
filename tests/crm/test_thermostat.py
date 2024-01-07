import torch
from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles

from conditional_rate_matching.models.generative_models.crm import CRM
from conditional_rate_matching.models.pipelines.thermostat.crm_thermostat_config import (
    LogThermostatConfig
)

from conditional_rate_matching.utils.integration import integrate_quad_tensor_vec

def test_constant_thermostat():
    configs = CRMConfig
    configs.thermostat = LogThermostatConfig()
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="trainer_call")
    config = CRMConfig()
    crm = CRM(config,experiment_files=experiment_files)
    batch_1, batch_0 = next(zip(crm.dataloader_1.train(), crm.dataloader_0.train()).__iter__())
    x1 = batch_1[0]
    x0 = batch_0[0]

    ta = torch.rand(x0.size(0)).to(crm.device)
    tb = torch.ones_like(ta).to(crm.device)


    result = integrate_quad_tensor_vec(crm.forward_rate.thermostat, ta, tb, 100)
    print("Cuadratures")
    print(result)
    analytical = crm.forward_rate.thermostat.integral(ta,tb)
    print("Anayltical")
    print(analytical)

def test_conditional_probability():
    configs = CRMConfig
    configs.thermostat = LogThermostatConfig()
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="trainer_call")
    config = CRMConfig()
    crm = CRM(config,experiment_files=experiment_files)
    batch_1, batch_0 = next(zip(crm.dataloader_1.train(), crm.dataloader_0.train()).__iter__())
    x1 = batch_1[0].to(crm.device)
    x0 = batch_0[0].to(crm.device)

    time = torch.rand(x0.size(0)).to(crm.device)
    x = crm.forward_rate.sample_x(x1,x0,time)
    conditional_rate = crm.forward_rate.conditional_transition_rate(x,x1,time)
    print(conditional_rate)







