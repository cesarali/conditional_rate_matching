import json

def store_metrics(config,all_metrics,new_metrics,metric_string_name,epoch,where_to_log=None):
    if where_to_log is None:
        mse_metric_path = config.experiment_files.metrics_file.format(metric_string_name + "_{0}_".format(epoch))
    else:
        mse_metric_path = where_to_log[metric_string_name]

    all_metrics.update(new_metrics)
    with open(mse_metric_path, "w") as f:
        json.dump(new_metrics, f)
    return all_metrics
