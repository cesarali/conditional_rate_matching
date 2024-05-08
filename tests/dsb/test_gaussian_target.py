from pprint import pprint

from conditional_rate_matching.configs.experiments_configs.dsb.dsb_experiments_graphs import experiment_comunity_small

from conditional_rate_matching.models.generative_models.dsb import DSB
from conditional_rate_matching.models.generative_models.dsb import DSBExperimentsFiles

def test_glauber_from_gm():
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="test",
                                           experiment_indentifier="test")
    config = experiment_comunity_small()
    pprint(config)
    dsb = DSB(config,experiment_files=experiment_files)


    databatch1 = next(dsb.dataloader_1.train().__iter__())
    x1 = databatch1[0].to(dsb.device)
    time_grid = dsb.pipeline.get_time_steps()
    path, time_grid = dsb.process.sample_path(x1,time_grid)