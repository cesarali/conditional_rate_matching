import numpy as np
from conditional_rate_matching.models.metrics import mmd

def marginal_histograms(generative_sample,test_sample):
    """

    :param oops:
    :param device:

    :return: backward_histogram,forward_histogram,forward_time

    """

    #================================
    # HISTOGRAMS OF DATA
    #================================
    marginal_histograms_data = test_sample.mean(axis=0).detach().cpu().numpy()
    marginal_histograms_sample = generative_sample.mean(axis=0).detach().cpu().numpy()
    #================================
    # HISTOGRAMS OF SAMPLE
    #================================
    mse = np.mean((marginal_histograms_data - marginal_histograms_sample)**2.)

    return mse

def kmmd(samples_0,sample_1):
    kmmd = mmd.MMD(mmd.scaled_exp_avg_hamming, False)
    opt_stat = kmmd.compute_mmd(samples_0.cpu(),sample_1.cpu())
    return opt_stat