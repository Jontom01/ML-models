from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from torch.utils.data import random_split, Subset
from base_models import SupervisedModel
from sklearn.model_selection import KFold
from torch import nn
import time
import torch

class HPSearch: #figure out how to make it so HPSearch can also take and properly sort other hyperparams that aren't just from the optimizer
    '''
    Performs hyperparameter on a search space using K-fold cross validation. Excepts a class implementation of HardClassifiers 
    '''
    def __init__(self, search_space, data, NetClass: SupervisedModel):
        self.search_space = search_space
        self.netclass = NetClass
        self.data = data

    def Search(self, num_folds=3, num_epochs=5, num_trials=6):
        trials = Trials()

        def _objective(params):
            K = num_folds
            kf = KFold(n_splits=K, shuffle=True, random_state=42)
            loss_val = 0

            for fold, (train_idx, val_idx) in enumerate(kf.split(self.data)):
                
                print(f"Fold: {fold}, Params: {params}")
                params_copy = params.copy()
                bs = params_copy.pop("batch_size", None) #this leaves just the optimizer params. MUST POP ANY NEW HYPERPARAMS THAT ARE NOT OPTIMIZER PARAMS
                if bs == None: bs = 32

                train_set = Subset(self.data, train_idx)
                val_set = Subset(self.data, val_idx)

                train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=False)

                net = self.netclass(optim_params=params_copy)
                net.fit(train_loader, epoch=num_epochs, verbose=False)
                eval_loss, _ = net.eval(val_loader)

                print(f"Evaluation of Fold: {fold}, Validation Loss: {eval_loss:.4f}")
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
            max_evals=num_trials,           #total number of trials
            trials=trials           #record of each trial’s results
        )

        print("Best hyperparameters:", best)

        for trial in trials.trials:
            print(trial["tid"], trial["result"]["loss"], trial["misc"]["vals"])
        return best