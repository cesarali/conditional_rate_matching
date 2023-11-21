import os
import unittest
from conditional_rate_matching.configs.config_files import get_experiment_dir
from conditional_rate_matching.models.generative_models.ctdd import CTDD
class TestLoading(unittest.TestCase):

    def test_loading(self):
        experiment_dir = get_experiment_dir(experiment_name="ctdd",experiment_type="graph",experiment_indentifier="community")
        ctdd =  CTDD(experiment_dir=experiment_dir)



if __name__=="__main__":
    unittest.main()