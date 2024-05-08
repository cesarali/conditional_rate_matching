from torch.utils.data import DataLoader
from image_datasets import load_nist_data
from utils import train_classifier
from architectures import LeNet5, ResNet18, ResNet34

#==================================
dataname = 'BinaryFashionMNIST'
network = 'LeNet5'
accuracy_goal = 0.95
device = 'cpu'
#==================================

if network == 'LeNet5': model = LeNet5(num_classes=27)
if network == 'ResNet18': model = ResNet18(num_classes=27)
if network == 'ResNet34': model = ResNet34(num_classes=27)

train = load_nist_data(name=dataname)
test = load_nist_data(name=dataname, train=False)
train_dataloader = DataLoader(train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test, batch_size=64, shuffle=False)

#...train classifier

train_classifier(model, 
                 train_dataloader=train_dataloader,
                 test_dataloader=test_dataloader,
                 device=device,
                 accuracy_goal=accuracy_goal*100,
                 lr=0.001,
                 max_epochs=10,
                 early_stopping=25,
                 save_as='models/{}_{}.pth'.format(network,'_'.join(dataname.split(' '))))