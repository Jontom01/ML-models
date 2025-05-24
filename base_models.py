from abc import ABC, abstractmethod
import torch
from torch import nn

class SupervisedModel(ABC):
    def __init__(self):
        self.loss = self._build_loss()
        self.net = self._build_net() #this has to be build before optim since optim requires the the NN params/weights
        self.optim = self._build_optim()

    @abstractmethod
    def _build_net(self):
        ...

    @abstractmethod
    def _build_optim(self):
        ...

    @abstractmethod
    def _build_loss(self):
        ...
        
    @abstractmethod
    def fit(self, loader, epoch, verbose):
        ...

    @abstractmethod
    def eval(self, loader):
        ...    

class HardClassifier(SupervisedModel):
    '''
    Base class for hard label classifiers using cross-entropy and SGD
    '''
    def __init__(self, optim_params, num_classes):
        self.optim_params = optim_params
        self.num_classes = num_classes

        super().__init__()

    @abstractmethod
    def _build_net(self):
        ...

    @abstractmethod
    def _build_optim(self):
        ...

    def _build_loss(self):
        return nn.CrossEntropyLoss()

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
