from conditional_rate_matching.configs.configs_classes.config_dsb import DSBConfig
from conditional_rate_matching.models.generative_models.dsb import DSB
from conditional_rate_matching.models.generative_models.dsb import DSBExperimentsFiles
from conditional_rate_matching.models.losses.dsb_losses_config import SteinSpinEstimatorConfig

def test_loss():
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="test",
                                           experiment_indentifier="test")
    config = DSBConfig()
    config.flip_estimator = SteinSpinEstimatorConfig()

    dsb = DSB(config,experiment_files=experiment_files)
    current_model = dsb.current_rate
    past_model = dsb.process
    sinkhorn_iteration = 0

    for X_spins,time in dsb.pipeline.sample_paths_for_training(past_model=past_model,sinkhorn_iteration=sinkhorn_iteration):
        loss = dsb.backward_ratio_estimator(current_model,
                                            past_model,
                                            X_spins,
                                            time,
                                            sinkhorn_iteration=sinkhorn_iteration)
        print(loss)
        break