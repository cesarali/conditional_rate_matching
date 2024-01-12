from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader

from conditional_rate_matching.models.generative_models.dsb import DSB
from conditional_rate_matching.models.generative_models.dsb import DSBExperimentsFiles

from conditional_rate_matching.configs.experiments_configs.dsb.dsb_experiments_graphs import experiment_comunity_small
from conditional_rate_matching.utils.plots.sb_plots import sinkhorn_plot

def test_pipeline():
    sinkhorn_iteration = 0
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="graph",
                                           experiment_indentifier="test")
    experiment_files.set_sinkhorn(sinkhorn_iteration)

    config = experiment_comunity_small()
    dsb = DSB(config,experiment_files=experiment_files)
    current_model = dsb.current_rate
    #past_model = dsb.past_rate
    past_model = dsb.process
    xf = dsb.pipeline(current_model,sample_size=29,forward=False)
    print(xf.shape)


def test_pipeline_log():
    sinkhorn_iteration = 0
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="graph",
                                           experiment_indentifier="test")
    experiment_files.set_sinkhorn(sinkhorn_iteration)

    config = experiment_comunity_small()
    dsb = DSB(config, experiment_files=experiment_files)
    current_model = dsb.current_rate
    #past_model = dsb.past_rate
    past_model = dsb.process

    config.trainer.max_test_size
    histogram0 = binary_histogram_dataloader(dsb.dataloader_0, dimensions=config.data0.dimensions,
                                             train=True, maximum_test_sample_size=1e10)
    histogram1 = binary_histogram_dataloader(dsb.dataloader_1, dimensions=config.data0.dimensions,
                                             train=True, maximum_test_sample_size=1e10)
    max_number_of_states_displayed = 8
    states_legends = list(map(str,list(range(max_number_of_states_displayed))))
    print(states_legends)

    backward_histogram,forward_histogram,forward_time = dsb.pipeline.histograms_paths_for_inference(current_model=current_model,
                                                                                                    past_model=past_model,
                                                                                                    sinkhorn_iteration=sinkhorn_iteration)


    sinkhorn_plot(sinkhorn_iteration,
                  histogram0,
                  histogram1,
                  backward_histogram,
                  forward_histogram,
                  forward_time,
                  states_legends,
                  max_number_of_states_displayed=8,
                  save_path=None)

def test_pipeline_for_training():
    sinkhorn_iteration = 0
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="graph",
                                           experiment_indentifier="test")
    experiment_files.set_sinkhorn(sinkhorn_iteration)

    config = experiment_comunity_small()
    dsb = DSB(config, experiment_files=experiment_files)
    current_model = dsb.current_rate
    #past_model = dsb.past_rate
    past_model = dsb.process
    sinkhorn_iteration = 0

    for x_path,time in dsb.pipeline.sample_paths_for_training(past_model=past_model,sinkhorn_iteration=sinkhorn_iteration):
        print(x_path.shape)
        print(time.shape)
        break

