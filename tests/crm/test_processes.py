import unittest

from conditional_rate_matching.configs.configs_classes.config_crm import CRMConfig

from conditional_rate_matching.models.pipelines.sdes_samplers.samplers import TauLeaping


class TestProcesses(unittest.TestCase):
    """
    """

    def test_k_processes(self):
        from conditional_rate_matching.data.dataloaders_utils import get_dataloaders

        config = CRMConfig()
        config.batch_size = 64

        # =====================================================
        # DATA STUFF
        # =====================================================
        dataloader_0,dataloader_1 = get_dataloaders(config)

        from conditional_rate_matching.models.generative_models.crm import conditional_transition_rate

        databatch_0,databatch_1 = next(zip(dataloader_0,dataloader_1).__iter__())
        x_0 = databatch_0[0]
        x_1 = databatch_1[0]

        #rate_model = lambda x, t: constant_rate(config, x, t)
        rate_model = lambda x, t: conditional_transition_rate(config, x, x_1, t)
        x_f, x_hist, x0_hist,ts = TauLeaping(config, rate_model, x_0, forward=True)
        print(x_hist.shape)

if __name__=="__main__":
    unittest.main()