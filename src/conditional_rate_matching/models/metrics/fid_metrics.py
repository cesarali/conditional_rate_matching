import os
import torch
from conditional_rate_matching.models.metrics.fid_nist.fid_metric import compute_fid
from conditional_rate_matching.models.metrics.fid_nist.fid_metric import compute_activation_statistics
from conditional_rate_matching.models.metrics.fid_nist.architectures import LeNet5
from conditional_rate_matching import project_path
from torch.utils.data.dataset import TensorDataset

fid_models_dir = os.path.join(project_path,
                              "src",
                              "conditional_rate_matching",
                              "models",
                              "metrics",
                              "fid_nist",
                              "models")
def load_classifier(dataset_name,device,fid_models_dir=fid_models_dir):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if dataset_name == "mnist":
        model = LeNet5(num_classes=10)
        model_path = os.path.join(fid_models_dir,'LeNet5_BinaryMNIST.pth')
        model.load_state_dict(torch.load(model_path))
    elif dataset_name == "emnist":
        model = LeNet5(num_classes=27)
        model_path = os.path.join(fid_models_dir,'LeNet5_BinaryEMNIST_Letters.pth')
        model.load_state_dict(torch.load(model_path))
    elif dataset_name == "fashion":
        model = LeNet5(num_classes=10)
        model_path = os.path.join(fid_models_dir,'LeNet5_BinaryFashionMNIST.pth')
        model.load_state_dict(torch.load(model_path))
    else:
        raise Exception
    model.eval()
    return model

def fid_nist(generative_sample,test_sample,dataset_name="mnist",device="cpu"):



    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if dataset_name == "mnist":
        model = LeNet5(num_classes=10)
        model_path = os.path.join(fid_models_dir,'LeNet5_BinaryMNIST.pth')
        model.load_state_dict(torch.load(model_path))
    elif dataset_name == "emnist":
        model = LeNet5(num_classes=27)
        model_path = os.path.join(fid_models_dir,'LeNet5_BinaryEMNIST_Letters.pth')
        model.load_state_dict(torch.load(model_path))
    elif dataset_name == "fashion":
        model = LeNet5(num_classes=10)
        model_path = os.path.join(fid_models_dir,'LeNet5_BinaryFashionMNIST.pth')
        model.load_state_dict(torch.load(model_path))
    else:
        raise Exception
    model.eval()

    test_dataset = TensorDataset(test_sample)
    generative_sample_dataset = TensorDataset(generative_sample)

    mu_1, sigma_1 = compute_activation_statistics(model, test_dataset, activation_layer='fc1', device=device)
    mu_2, sigma_2 = compute_activation_statistics(model, test_dataset, activation_layer='fc2', device=device)
    mu_3, sigma_3 = compute_activation_statistics(model, test_dataset, activation_layer='fc3', device=device)

    fid_1 = compute_fid(model, generative_sample_dataset, mu_ref=mu_1, sigma_ref=sigma_1, activation_layer='fc1', device=device)
    fid_2 = compute_fid(model, generative_sample_dataset, mu_ref=mu_2, sigma_ref=sigma_2, activation_layer='fc2', device=device)
    fid_3 = compute_fid(model, generative_sample_dataset, mu_ref=mu_3, sigma_ref=sigma_3, activation_layer='fc3', device=device)

    return {"fid_1":fid_1.item(),"fid_2":fid_2.item(),"fid_3":fid_3.item()}


