import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import random_split, Subset
from hyperopt import hp, space_eval
from base import LinearRegressor
from param_search_utils import HPSearch
import matplotlib.pyplot as plt
import seaborn as sns

class MLPRegressor(LinearRegressor):
    '''
    Multilayer linear regressor
    '''
    def __init__(self, optim_params={ #these params are best with batch_size=128
        "lr": 1.78e-5,
    },
    net_params={
        "num_features": 2
    }
    ):
        super().__init__(optim_params=optim_params, net_params=net_params)

    def _build_net(self):
        return nn.Sequential(
            nn.Linear(in_features=self.net_params["num_features"], out_features=512), nn.ReLU(),
            nn.Linear(in_features=512, out_features=256), nn.ReLU(),
            nn.Linear(in_features=256, out_features=64), nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
            )
    
    def _build_optim(self):
        return torch.optim.AdamW(self.net.parameters(), **self.optim_params)

def student_performance_dataset():
    #Preprocess dataset
    df = pd.read_csv("./datasets/lin_reg_kaggle/Student_Performance.csv")
    label = "Performance Index"

    df = df.drop(columns=["Extracurricular Activities", "Sleep Hours"]) #sns.pairwise showed no pattern between these 2 cols and the performance index
    df = df[df.columns].apply(lambda x: (x - x.mean()) / (x.std()))

    #DataFrame -> Pytorch ready
    X_pre = df.drop(columns=[label]).values
    y_pre = df[label].values

    X = torch.tensor(X_pre, dtype=torch.float32)
    y = torch.tensor(y_pre, dtype=torch.float32)
    y = y.unsqueeze(1) #so that the shape between my model output and the actual label shape are the same

    dataset = torch.utils.data.TensorDataset(X, y)

    return dataset

if __name__ == "__main__":

    #Create datasets
    dataset = student_performance_dataset()
    train_set = Subset(dataset, list(range(7000)))
    test_set = Subset(dataset, list(range(7000, len(dataset))))
    
    #Pairwise plot
    #df = pd.read_csv("./datasets/lin_reg_kaggle/Student_Performance.csv")
    ##sns.pairplot(df)
    #plt.show()

    '''
    #HYPER PARAM SEARCH
    search_space = {
        #"weight_decay": hp.loguniform("weight_decay", np.log(1e-5), np.log(1e-3)),
        "lr":         hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
        "batch_size": hp.choice("batch_size", [128, 256, 512, 1024]),
        #"momentum": hp.uniform("momentum", 0.5, 0.99)
    }

    param_search = HPSearch(search_space=search_space, NetClass=MLPRegressor, data=train_set)

    best_params = param_search.Search(num_folds=4, num_epochs=30, num_trials=20) #index of the best params

    best_params = space_eval(search_space, best_params) #get the actual values

    batch_size = best_params.pop("batch_size") #remove batch size since its not an optim_param
    
    
    #TRAINING AND SAVING A MODEL USING THE BEST HYPERPARAMS
    linreg = MLPRegressor(optim_params=best_params)

    linreg.fit(loader=torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True), epoch=100, verbose=True)

    loss = linreg.eval(loader=torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False), verbose=True)
    print(f"Test set loss: {loss}")

    torch.save(linreg.net.state_dict(), "./supervised_learning/models/student_performance_regressor_3features.pth")
    
    '''

    #RUN BEST MODEL
    linreg = MLPRegressor(optim_params={ "lr": 8.5248e-05}, net_params={"num_features": 3}) #AdamW optimizer
    linreg.net.load_state_dict(torch.load("./supervised_learning/models/student_performance_regressor_3features.pth"))
    loss = linreg.eval(loader=torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False), verbose=True)
    print(f"Test set loss: {loss}")
    
    