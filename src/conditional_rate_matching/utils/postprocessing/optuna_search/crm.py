import optuna
from conditional_rate_matching.configs.config_crm import CRMConfig
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.models.trainers.crm_trainer import CRMTrainer
from experiments.testing_graphs import small_community, community


def objective(trial):
    # Generate the hyperparameters
    time_embed_dim = trial.suggest_int('time_embed_dim', 50, 100)
    hidden_dim = trial.suggest_int('hidden_dim', 50, 100)

    # Files to save the experiments_configs
    experiment_files = ExperimentFiles(experiment_name="crm",
                                       experiment_type="optuna_search2",
                                       experiment_indentifier=None)

    # Update the config
    config = small_community(number_of_epochs=5)

    config.model_mlp.time_embed_dim = time_embed_dim
    config.model_mlp.hidden_dim = hidden_dim

    #config.process.gamma = 0.1

    # Assuming you have a function to train your model and return a metric
    # For example: train_model(config) -> returns accuracy or loss
    crm_trainer = CRMTrainer(config,experiment_files)
    results_,all_metrics = crm_trainer.train()
    metric = all_metrics["mse_marginal_histograms"]

    return metric

if __name__=="__main__":
    study = optuna.create_study(direction='minimize',storage='sqlite:///example.db')  # or 'minimize' if you are minimizing a metric like loss
    study.optimize(objective, n_trials=5)