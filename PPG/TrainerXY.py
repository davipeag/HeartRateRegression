

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Callable
import copy


class BatchTraninerXY():
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.net = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    @property
    def model(self):
        return self.net

    def train(self, batch):
        self.net.train()
        self.net.zero_grad()
        x, y, p = self.compute(batch)
        loss = self.criterion(p, y)
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def compute(self, batch):
        x, y = map(lambda v: v.to(self.device), batch)
        p = self.net(x)
        return x, y, p

    def evaluate(self, batch):
        self.net.eval()
        with torch.no_grad():
            return [v.detach().cpu() for v in self.compute(batch)]

    def validate(self, batch):
        x, y, p = self.evaluate(batch)
        return self.criterion(y, p).cpu().item()


class EpochTrainerXY():
    def __init__(self, model, optimizer, criterion, device="cpu"):
        self.net = model
        self.trainer = BatchTraninerXY(model, optimizer, criterion, device)

    @property
    def model(self):
        return self.net

    def apply_to_batches(self, function, loader):
        return [function(batch) for batch in loader]

    def train(self, loader):
        return self.apply_to_batches(self.trainer.train, loader)

    def validate(self, loader):
        self.apply_to_batches(self.trainer.validate, loader)

    def evaluate(self, loader):
        return [torch.cat(l) for l in zip(*self.apply_to_batches(self.trainer.evaluate, loader))]


class MetricsComputerXY():
    def __init__(self, ztransformer, epoch_trainer: EpochTrainer):
        self.ztransformer

    def inverse_transform_label(self, y):

        ymean = self.ztransformer.mean_[0]
        ystd = self.ztransformer.scale_[0]

        return (y*ystd)+ymean

    def plot_predictions(self, y, predictions, print_indices=[0]):

        yp = y.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(yp)
        pr = self.inverse_transform_label(p)

        for i in print_indices:
            plt.figure()
            plt.plot(np.linspace(
                1, 8*yr.shape[1]-1, yr.shape[1]), yr[i], 'b', label="label")
            plt.plot(np.linspace(
                1, 8*yr.shape[1]-1, yr.shape[1]), pr[i], 'k', label="predictions")
            plt.xlabel("seconds")
            plt.legend()
            plt.show()

    def mae(self, yp, predictions):
        # yi = yi.detach().cpu().numpy()
        yp = yp.detach().cpu().numpy()
        p = predictions.detach().cpu().numpy()

        yr = self.inverse_transform_label(yp)
        pr = self.inverse_transform_label(p)

        return np.abs(yr-pr).mean()

    def rmse(self, y, p):
        yr = self.inverse_transform_label(y.detach().cpu().numpy())
        pr = self.inverse_transform_label(p.detach().cpu().numpy())

        return (((yr-pr)**2).mean())**0.5


class TrainHelperXY():
    def __init__(self, trainer: EpochTrainerXY, loader_tr, loader_val, loader_ts, display_criterion: Callable[[torch.Tensor, torch.Tensor], float]):
        self.trainer = trainer
        self.loader_tr = loader_tr
        self.loader_ts = loader_ts
        self.loader_val = loader_val
        self.display_criterion = display_criterion

    def train(self, nepoch):
        loader_tr, loader_val, loader_ts = make_loaders(ts_sub, val_sub)

    def compute_metric(self, loader):
        x,y,p = self.trainer.evaluate(loader)
        return self.display_criterion(y,p)
    
    def train(nepoch):
        best_val_model = copy.deepcopy(self.trainer.model.state_dict())
        train_metrics = []
        validation_metrics, test_metrics = [[self.compute_metric(l)] for l in [loader_val, loader_ts]] 
    
    
        for epoch in range(1, n_epoch+1):
            self.trainer.train(self.loader_tr)
            loss_tr = self.compute_metric(self.loader_tr)
            loss_val = self.compute_metric(self.loader_val)
            loss_ts = self.compute_metric(self.loader_ts)
            
            if val_loss < np.min(validation_metrics):
                print("best val epoch:", epoch)
                best_val_model = copy.deepcopy(net.state_dict())
            print('[%d/%d]: loss_train: %.3f loss_val %.3f loss_ts %.3f' % (
                  (epoch), n_epoch, loss_ts, loss_val, loss_ts))

            train_metrics.append(loss_tr)
            validation_metrics.append(loss_val)
            test_metrics.append(loss_ts)
        
        self.trainer.model.load_state_dict(best_val_model)
        idx = validation_metrics.index(np.min(validation_metrics)) 
        return test_metrics[idx]
            
            

        print(f"Final: {val_sub}-{ts_sub}:{compute_mean_MAE(loader_ts_long)}\n####")

