import torch
import numpy as np
from torch import nn
from torch.utils.data import random_split
from hyperopt import hp
from base_models import LinearRegressor
from param_search_utils import HPSearch

class BasicRegressor(LinearRegressor):
    '''
    Simple in-and-out linear regressor
    '''
    def __init__(self, optim_params={
        "lr": 1e-2, 
        "momentum": 0.745, 
        "weight_decay": 1e-4
    },
    net_params={
        "num_features": 2
    }
    ):
        super().__init__(optim_params=optim_params, net_params=net_params)

    def _build_net(self):
        return nn.Linear(in_features=self.net_params["num_features"], out_features=1)
    
    def _build_optim(self):
        return torch.optim.SGD(self.net.parameters(), **self.optim_params)

if __name__ == "__main__":

    #Synthetic data
    torch.manual_seed(0)
    n_samples, n_features = 1000, 2
    X = torch.randn(n_samples, n_features)
    w_true = torch.tensor([[2.0], [-3.4]])
    b_true = 4.2
    noise = torch.randn(n_samples, 1) * 0.01
    y = X @ w_true + b_true + noise   

    X_train = X[:800]
    y_train = y[:800]
    X_test   = X[800:]
    y_test   = y[800:]

    #HYPER PARAM SEARCH
    search_space = {
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-5), np.log(1e-3)),
        "lr":         hp.loguniform("lr", np.log(1e-3), np.log(1e-1)),
        "batch_size": hp.choice("batch_size", [32, 64]),
        "momentum": hp.uniform("momentum", 0.5, 0.99)
    }
    dataset = torch.utils.data.TensorDataset(X_train, y_train)

    param_search = HPSearch(search_space=search_space, NetClass=BasicRegressor, data=dataset)

    param_search.Search(num_folds=20, num_epochs=10)