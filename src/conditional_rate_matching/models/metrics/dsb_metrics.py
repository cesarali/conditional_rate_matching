from conditional_rate_matching.models.metrics.histograms import binary_histogram_dataloader
from conditional_rate_matching.utils.plots.sb_plots import sinkhorn_plot

def sb_plot(dsb,current_model,past_model,sinkhorn_iteration,save_path=None,max_number_of_states_displayed = 8):
    config = dsb.config
    histogram0 = binary_histogram_dataloader(dsb.dataloader_0, dimensions=config.data0.dimensions,
                                             train=True, maximum_test_sample_size=1e10)
    histogram1 = binary_histogram_dataloader(dsb.dataloader_1, dimensions=config.data0.dimensions,
                                             train=True, maximum_test_sample_size=1e10)

    states_legends = list(map(str, list(range(max_number_of_states_displayed))))

    backward_histogram, forward_histogram, forward_time = dsb.pipeline.histograms_paths_for_inference(
        current_model=current_model,
        past_model=past_model,
        sinkhorn_iteration=sinkhorn_iteration)

    sinkhorn_plot(sinkhorn_iteration,
                  histogram0,
                  histogram1,
                  backward_histogram,
                  forward_histogram,
                  forward_time,
                  states_legends,
                  max_number_of_states_displayed=max_number_of_states_displayed,
                  save_path=save_path)