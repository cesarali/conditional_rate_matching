## __EXPERIMENTS__

We designed experiments for 4 main datasets:

+ Graph Networks
+ MNIST Binarized
+ Swiss Roll
+ Spin Glass Systems

The configs are defined through classes obtained in the folder "config classes", see below: 

```
...
├── src
│   └── conditional_rate_matching        <- Actual Python package where the main functionality goes.
...
│       ├── models
│           ├── metrics
│               ├── orca                   <- compile orca here 
│       ├── configs
│           ├── configs_classes            <- Classes for the configurations
│           ├── experiments_configs        <- Here the configurations are desined per experiments with different functions
...
├── data
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
```

Constructions of different configs files as required for the experiments are obtained in 
"experiments_configs". One can directly run a training procedure by calling:

python ./experiments_configs/crm/crm_experiments_graphs.py

For the graph experiments, if you want to include the graph metrics you have to have compiled the orca at:
/src/conditional_rate_matching/metrics/orca

### comment this line
This is since other users locate the orca at different folders, by locating in /src/conditional_rate_matching/metrics/orca we expect linux

config.trainer.orca_dir = None

### uncomment this line:
#config.trainer.metrics.append(MetricsAvaliable.graphs_metrics)



