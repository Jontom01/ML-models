import torch
import torchvision
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import random_split
from hyperopt import hp
from base import HardClassifier
from param_search_utils import HPSearch

class LeNet(HardClassifier):
    '''
    Slightly modernized version of the LeNet CNN
    '''
    def __init__(self, optim_params={
        "lr": 0.0184, 
        "momentum": 0.8546, 
        "weight_decay": 1e-3
    },
    net_params={
        "num_classes": 10
    }):
        super().__init__(optim_params=optim_params, net_params=net_params)

    def _build_net(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.Linear(in_features=800, out_features=256), nn.ReLU(), 
            nn.Linear(in_features=256, out_features=128), nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.net_params["num_classes"])
        )
    
    def _build_optim(self):
        return torch.optim.SGD(self.net.parameters(), **self.optim_params)

class MLPClassifier(HardClassifier):
    '''
    Simple MLP Classifier
    '''
    def __init__(self, optim_params={
        "lr": 1e-2, 
        "momentum": 0.55, 
        "weight_decay": 1e-4
    },
    net_params={
        "num_classes": 10
    }):
        super().__init__(optim_params=optim_params, net_params=net_params)

    def _build_net(self):
        return nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=784, out_features=256), nn.ReLU(), nn.Dropout(0.2), 
            nn.Linear(in_features=256, out_features=128), nn.ReLU(), nn.Dropout(0.1), 
            nn.Linear(in_features=128, out_features=64), nn.ReLU(), nn.Dropout(0.05), 
            nn.Linear(in_features=64, out_features=self.net_params["num_classes"])
        )
    
    def _build_optim(self):
        return torch.optim.SGD(self.net.parameters(), **self.optim_params)

def get_FashionMNIST(): #eventually turn this into a class that'll generalize to all pytorch downloadable image datasets
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((28,28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    train_data = torchvision.datasets.FashionMNIST(root="./datasets", train=True, transform=trans, download=True)
    test_data = torchvision.datasets.FashionMNIST(root="./datasets", train=False, transform=trans, download=True)
    return train_data, test_data

if __name__ == "__main__":
    full_trainset, test_set = get_FashionMNIST()
    train_set, val_set = random_split(full_trainset, [50000, 10000])
    '''
    #HYPER PARAM SEARCH
    search_space = {
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-3), np.log(1.01e-3)),
        "lr":         hp.loguniform("lr", np.log(0.0184), np.log(0.0185)),
        "batch_size": hp.choice("batch_size", [32, 64, 128, 256]),
        "momentum": hp.uniform("momentum", 0.854, 0.855)
    }
    
    param_search = HPSearch(search_space=search_space, NetClass=MLPClassifier, data=full_trainset)

    param_search.Search()
    '''
    #LOSS/ACCURACY TEST

    leNet = LeNet()

    leNet.fit(loader=torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True), epoch=20, verbose=True)

    loss, accuracy = leNet.eval(loader=torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False))
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    #torch.save(leNet.net.state_dict(), "./supervised_learning/models/modified_LeNet_classifier.pth")