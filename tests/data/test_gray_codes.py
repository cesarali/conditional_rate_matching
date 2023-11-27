import os
import torch
import unittest
from conditional_rate_matching.data.image_dataloader_config import NISTLoaderConfig
from conditional_rate_matching.data.image_dataloaders import NISTLoader
from conditional_rate_matching.utils.data import check_sizes

from conditional_rate_matching.data.gray_codes_dataloaders_config import GrayCodesDataloaderConfig, AvailableGrayCodes
from conditional_rate_matching.data.gray_codes_dataloaders import OnlineToyDataset
from conditional_rate_matching.data.gray_codes_dataloaders import get_true_samples
from conditional_rate_matching.utils.plots.gray_code_plots import get_binmap
from conditional_rate_matching.configs.config_files import ExperimentFiles
from conditional_rate_matching.utils.plots.gray_code_plots import bin2float,plot_samples
class TestGrayCodesDataLoader(unittest.TestCase):

    def test_gray_codes(self):
        data_config = GrayCodesDataloaderConfig(batch_size=23,dataset_name=AvailableGrayCodes.checkerboard)

        experiment_files = ExperimentFiles(experiment_name="gray",
                                           experiment_type="plot")
        experiment_files.create_directories()
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        print("Schon")
        ############## Data
        discrete_dim = 32
        bm, inv_bm = get_binmap(discrete_dim, 'gray')

        db = OnlineToyDataset(data_config.dataset_name, discrete_dim)
        plot_size = db.f_scale
        int_scale = db.int_scale

        batch_size = data_config.batch_size
        multiples = {'pinwheel': 5, '2spirals': 2}
        batch_size = batch_size - batch_size % multiples.get(data_config.dataset_name, 1)
        x = get_true_samples(db, batch_size, bm, int_scale, discrete_dim).to(device)
        print(x.shape)

        # samples of gfn
        gfn_samp_float = bin2float(x.data.cpu().numpy().astype(int), inv_bm, int_scale, discrete_dim)
        plot_samples(gfn_samp_float, experiment_files.plot_path.format("gray_scale_{}"), lim=plot_size)

if __name__=="__main__":
    unittest.main()