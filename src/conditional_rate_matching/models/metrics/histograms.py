import torch
import torch.nn.functional as F

def categorical_histogram_dataloader(dataloader_0, dimensions, number_of_classes, train=True,maximum_test_sample_size = 2000):
    """
    Just the marginal per dimension

    :param dataloader_0:
    :param dimensions:
    :param number_of_classes:
    :param train:
    :return:
    """
    if hasattr(dataloader_0, "train"):
        if train:
            dataloader = dataloader_0.train()
        else:
            dataloader = dataloader_0.test()
    else:
        dataloader = dataloader_0

    histogram = torch.zeros(dimensions,number_of_classes)
    sample_size = 0.
    for databatch in dataloader:
        x_0 = databatch[0]
        sample_size += x_0.size(0)
        histogram += F.one_hot(x_0.long(),num_classes=number_of_classes).sum(axis=0)
        if sample_size > maximum_test_sample_size:
            break
    histogram = histogram / sample_size
    return histogram


def binary_histogram_dataloader(dataloader_0, dimensions, train=True, maximum_test_sample_size=2000):
    """
    Just the marginal per dimension

    :param dataloader_0:
    :param dimensions:
    :param number_of_classes:
    :param train:
    :return:
    """
    histogram = categorical_histogram_dataloader(dataloader_0, dimensions,
                                                 number_of_classes=2, train=train,
                                                 maximum_test_sample_size=maximum_test_sample_size)
    assert histogram.size(1) == 2
    return histogram[:,1]

