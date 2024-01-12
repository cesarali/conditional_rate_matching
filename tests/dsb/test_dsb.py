from conditional_rate_matching.models.generative_models.dsb import DSB
from conditional_rate_matching.models.generative_models.dsb import DSBExperimentsFiles

def test_load_model():
    experiment_files = DSBExperimentsFiles(experiment_name="dsb",
                                           experiment_type="graph",
                                           experiment_indentifier="training_test",
                                           delete=True)
    dsb = DSB(experiment_dir=experiment_files.experiment_dir,sinkhorn_iteration=4)
