import time
import torch
import torchvision
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import random_split, Subset
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from abc import ABC, abstractmethod

class HardClassifier(ABC):
    '''
    Base class for hard label classifiers using cross-entropy and SGD
    '''
    def __init__(self, optim_params, num_classes):
        self.optim_params = optim_params
        self.num_classes = num_classes

        self.loss = nn.CrossEntropyLoss()
        self.net = self._build_net()
        self.optim = self._build_optim()

    @abstractmethod
    def _build_net(self):
        ...

    @abstractmethod
    def _build_optim(self):
        ...

    def fit(self, loader, epoch=20, verbose=True):
        self.net.train()
        for i in range(epoch):
            loss_epoch = 0
            for X, y in loader:
                self.optim.zero_grad()
                loss_ret = self.loss(self.net(X), y)
                loss_epoch += loss_ret.item()
                loss_ret.backward()
                self.optim.step()
            if verbose:
                print(f"epoch {i} loss: ", loss_epoch / len(loader))

    def eval(self, loader):
        self.net.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in loader:
                logits = self.net(X)
                loss = self.loss(logits, y)
                total_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

class LeNet(HardClassifier):
    '''
    Slightly modernized version of the LeNet CNN
    '''
    def __init__(self, optim_params={
        "lr": 0.0184, 
        "momentum": 0.8546, 
        "weight_decay": 1e-3
    },
    num_classes=10):
        super().__init__(optim_params=optim_params, num_classes=num_classes)

    def _build_net(self):
        return nn.Sequential(
            nn.LazyConv2d(12, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(32, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(), nn.LazyLinear(800, 256), nn.ReLU(), 
            nn.LazyLinear(256, 128), nn.ReLU(),
            nn.LazyLinear(128, self.num_classes)
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
    num_classes=10):
        super().__init__(optim_params=optim_params, num_classes=num_classes)

    def _build_net(self):
        return nn.Sequential(
            nn.Flatten(), nn.LazyLinear(784, 256), nn.ReLU(), 
            nn.Dropout(0.2), nn.LazyLinear(256, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.LazyLinear(128, 64), nn.ReLU(),
            nn.Dropout(0.05), nn.LazyLinear(64, self.num_classes)
        )
    
    def _build_optim(self):
        return torch.optim.SGD(self.net.parameters(), **self.optim_params)

class HPSearch:
    '''
    Performs hyperparameter on a search space using K-fold cross validation. Excepts a class implementation of HardClassifiers 
    '''
    def __init__(self, search_space, data, NetClass: HardClassifier):
        self.search_space = search_space
        self.netclass = NetClass
        self.data = data

    def Search(self):
        trials = Trials()

        def _objective(params):
            K = 3
            kf = KFold(n_splits=K, shuffle=True, random_state=42)
            loss_val = 0

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.data)):
                
                print(f"Fold: {fold}, Params: {params}")
                
                bs = params.pop("batch_size", None) #this leaves just the optimizer params. MUST POP ANY NEW HYPERPARAMS THAT ARE NOT OPTIMIZER PARAMS
                if bs == None: bs = 32

                train_set = Subset(self.data, train_idx)
                val_set = Subset(self.data, val_idx)

                train_loader = get_dataloader(train_set, batch_size=bs, train=True)
                val_loader = get_dataloader(val_set, batch_size=bs, train=False)

                net = self.netclass(optim_params=params, num_classes=10)
                
                net.fit(train_loader, epoch=5)
                eval_loss, accuracy = net.eval(val_loader)

                print(f"Evaluation of Fold: {fold}, Validation Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")
                loss_val += eval_loss

            return {
            "loss":      loss_val / K,       #the quantity to minimize
            "status":    STATUS_OK,      #always OK (or STATUS_FAIL)
            #optional: can return other metrics for logging
            "eval_time": time.time()     
            }
        
        best = fmin(
            fn=_objective,           #objective function
            space=self.search_space,     #the hyperparameter space
            algo=tpe.suggest,       #the Tree-structured Parzen Estimator
            max_evals=6,           #total number of trials
            trials=trials           #record of each trial’s results
        )

        print("Best hyperparameters:", best)

        for trial in trials.trials:
            print(trial["tid"], trial["result"]["loss"], trial["misc"]["vals"])

def get_FashionMNIST():
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((28,28)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    train_data = torchvision.datasets.FashionMNIST(root="./my_data_dir", train=True, transform=trans, download=True)
    test_data = torchvision.datasets.FashionMNIST(root="./my_data_dir", train=False, transform=trans, download=True)
    return train_data, test_data

def get_dataloader(data, batch_size, train=False):
    return torch.utils.data.DataLoader(data, batch_size, shuffle=train)

if __name__ == "__main__":

    search_space = {
        # continuous uniform between 0 and 1
        #"dropout_1":    hp.uniform("dropout_1", 0.0, 0.5),
        #"dropout_2":    hp.uniform("dropout_2", 0.0, 0.5),
        #"dropout_3":    hp.uniform("dropout_3", 0.0, 0.5),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-3), np.log(1.01e-3)),
        # log–uniform learning‐rate between 1e–5 and 1e–1
        "lr":         hp.loguniform("lr", np.log(0.0184), np.log(0.0185)),
        # choice among a few options
        "batch_size": hp.choice("batch_size", [32, 64, 128, 256]),
        # conditional subspace: only pick momentum if using SGD
        #"optimizer":  hp.choice("opt", [
        #    {"type": "sgd", "momentum": hp.uniform("momentum", 0.5, 0.99)},
        #    {"type": "adam"}  # no extra params
        #])
        "momentum": hp.uniform("momentum", 0.854, 0.855)
    }
    
    #HYPER PARAM SEARCH
    full_trainset, test_set = get_FashionMNIST()
    train_set, val_set = random_split(full_trainset, [50000, 10000])
    
    param_search = HPSearch(search_space=search_space, NetClass=MLPClassifier, data=full_trainset)

    param_search.Search()
    '''
    #SIMPLE LOSS/ACCURACY TEST
    leNet = LeNet()
    mlp = MLPClassifier()

    full_trainset, test_set = get_FashionMNIST()
    train_set, val_set = random_split(full_trainset, [50000, 10000])
    train_loader = get_dataloader(train_set, batch_size=32, train=True)
    val_loader = get_dataloader(val_set, batch_size=32, train=False)

    leNet.fit(train_loader, epoch=10)
    eval_loss, accuracy = leNet.eval(val_loader)

    print(f"Validation Loss: {eval_loss:.4f}, Accuracy: {accuracy:.4f}")
    '''
    #torch.save(leNet.net.state_dict(), "LeNet_edited2_test.pth")