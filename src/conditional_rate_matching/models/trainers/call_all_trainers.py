from typing import Union

from conditional_rate_matching.configs.config_files import ExperimentFiles

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig
from conditional_rate_matching.configs.configs_classes.config_ctdd import CTDDConfig
from conditional_rate_matching.configs.config_oops import OopsConfig

from conditional_rate_matching.models.trainers.ctdd_trainer import CTDDTrainer
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from conditional_rate_matching.models.trainers.oops_trainer import OopsTrainer

def call_trainer(config:Union[CRMConfig,CTDDConfig,OopsConfig],experiment_name="general_call"):
    if isinstance(config,CRMConfig):
        experiment_files = ExperimentFiles(experiment_name=experiment_name,experiment_type="trainer_call")
        crm_trainer = CRMTrainer(config, experiment_files)
        results_, all_metrics = crm_trainer.train()
    elif isinstance(config,CTDDConfig):
        experiment_files = ExperimentFiles(experiment_name=experiment_name,experiment_type="trainer_call")
        ctdd_trainer = CTDDTrainer(config, experiment_files)
        results_, all_metrics = ctdd_trainer.train()
    elif isinstance(config,OopsConfig):
        experiment_files = ExperimentFiles(experiment_name=experiment_name,experiment_type="trainer_call")
        oops_trainer = OopsTrainer(config, experiment_files)
        results_, all_metrics = oops_trainer.train()
    else:
        results_, all_metrics = None,None
    return results_, all_metrics