import os
from conditional_rate_matching.configs.experiments_configs.crm.crm_experiments_graph import experiment_comunity_small
from conditional_rate_matching.models.generative_models.crm import CRM

from pprint import pprint
from dataclasses import asdict

if __name__=="__main__":
    graph_config = experiment_comunity_small()
    #pprint(asdict(graph_config))

    crm = CRM(config=graph_config)
    databatch = next(crm.dataloader_1.train().__iter__())
    x = databatch[0]
    
    print(databatch)

