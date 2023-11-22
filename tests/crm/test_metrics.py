import unittest

from conditional_rate_matching.configs.config_crm import CRMConfig as ConditionalRateMatchingConfig

from conditional_rate_matching.data.dataloaders_utils import get_dataloaders
from conditional_rate_matching.utils.plots.histograms_plots import kHistogramPlot
from conditional_rate_matching.models.metrics.histograms import categorical_histogram_dataloader

@unittest.skip
class TestHistograms(unittest.TestCase):
    """

    """
    def test_dataloaders_histograms(self):
        config = ConditionalRateMatchingConfig(batch_size=128,sample_size=1000)
        dataloader_0,dataloader_1 = get_dataloaders(config)
        histogram = categorical_histogram_dataloader(dataloader_0, config.data1.dimensions, config.data1.vocab_size)
        kHistogramPlot(config, histogram, t=0)
        histogram = categorical_histogram_dataloader(dataloader_1, config.data1.dimensions, config.data1.vocab_size)
        kHistogramPlot(config, histogram, t=1)


class TestGraphsMetrics(unittest.TestCase):
    """
    """
    def test_graph(self):
        from experiments.testing_graphs import small_community
        from conditional_rate_matching.models.generative_models.crm import CRM
        from conditional_rate_matching.models.pipelines.sdes_samplers.samplers_utils import sample_from_dataloader
        from conditional_rate_matching.models.metrics.graphs_metrics import eval_graph_list

        config = small_community()
        crm = CRM(config=config)
        dataloader = crm.dataloader_1.test()
        test_sample = sample_from_dataloader(dataloader, sample_size=20).to(crm.device)
        generative_sample, generative_path, ts = crm.pipeline(sample_size=test_sample.shape[0],
                                                              return_intermediaries=True,
                                                              train=False)
        test_graphs = crm.dataloader_1.sample_to_graph(test_sample)
        generative_graphs = crm.dataloader_1.sample_to_graph(generative_sample)

        #plot_graphs_list2(generative_graphs,title="Generative")
        #plot_graphs_list2(test_graphs,title="Test Graphs")

        graph_metrics = eval_graph_list(generative_graphs,test_graphs)
        #print(graph_metrics)
        #print(test_sample.shape)
        #print(generative_sample)



if __name__=="__main__":
    unittest.main()



