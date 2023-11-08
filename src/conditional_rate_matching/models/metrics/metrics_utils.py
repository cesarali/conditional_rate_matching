import json
key_in_dict = lambda dictionary, key: dictionary is not None and key in dictionary


def store_metrics(experiment_files,all_metrics,new_metrics,metric_string_name,epoch,where_to_log=None):
    if key_in_dict(where_to_log, metric_string_name):
        mse_metric_path = where_to_log[metric_string_name]
    else:
        mse_metric_path = experiment_files.metrics_file.format(metric_string_name + "_{0}_".format(epoch))
    all_metrics.update(new_metrics)
    with open(mse_metric_path, "w") as f:
        json.dump(new_metrics, f)
    return all_metrics
