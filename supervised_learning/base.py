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
    Base class for hard label classifiers using cross-entropy loss
    '''
    def __init__(self, optim_params, net_params):
        self.optim_params = optim_params
        self.net_params = net_params

        super().__init__()

    @abstractmethod
    def _build_net(self):
        ...

    @abstractmethod
    def _build_optim(self):
        ...

    def _build_loss(self):
        return nn.CrossEntropyLoss()

    def fit(self, loader, epoch=20, verbose=False):
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

class LinearRegressor(SupervisedModel):
    '''
    Base class for linear regression models using MSE loss
    '''
    def __init__(self, optim_params, net_params):
        self.optim_params = optim_params
        self.net_params = net_params

        super().__init__()

    @abstractmethod
    def _build_net(self):
        ...

    @abstractmethod
    def _build_optim(self):
        ...
    
    def _build_loss(self):
        return nn.MSELoss()

    def fit(self, loader, epoch=20, verbose=False):
        self.net.train()
        for i in range(epoch):
            loss_epoch = 0
            for X, y in loader: #each batch
                y_pred = self.net(X)
                loss_ret = self.loss(y_pred, y)
                loss_epoch += loss_ret.item()
                self.optim.zero_grad()
                loss_ret.backward()
                self.optim.step()
            if verbose:
                print(f'Epoch {i}, Loss: {loss_ret.item():.4f}')
    
    def eval(self, loader, verbose=False):
        self.net.eval()
        total_loss, total = 0.0, 0
        with torch.no_grad():
            for X, y in loader: #each batch
                y_pred = self.net(X)
                if verbose: #mean percent error. However, take note that this is the mean percent error of whatever form your data currently is in
                    pct_err = torch.abs((y_pred - y) / (y + 1e-6)) * 100
                    mean_pct_err = pct_err.mean().item()
                    print(f"Mean % error = {mean_pct_err}%")
                total_loss += self.loss(y_pred, y).item() * X.size(0)
                total += y.size(0)
        avg_loss = total_loss / total

        return avg_loss, None