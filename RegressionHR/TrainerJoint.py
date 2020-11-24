

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Callable
import copy

class LossComputer():
    def __init__(self, model, criterion, device="cpu"):
        self.net = model
        self.criterion = criterion
        self.device = device

    @property
    def model(self):
        return self.net

    def compute(self, batch):
        batch_d = list(map(lambda v: v.to(self.device), batch))
        p = self.net(*batch_d[:-1])
        return [*batch_d, p]

    def compute_loss(self, batch):
        y,p = self.compute(batch)[-2:]
        return self.criterion(p, y)

    def evaluate(self, batch):
        self.net.eval()
        with torch.no_grad():
            return [v.detach().cpu() for v in self.compute(batch)]

    def validate(self, batch):
        y, p = self.evaluate(batch)[-2:]
        return self.criterion(p, y).cpu().item()


class BatchTraninerJoint():
    def __init__(self, loss_computer1: LossComputer, loss_computer2: LossComputer, optimizer, alpha, device="cpu"):
        self.optimizer = optimizer
        self.loss_computers = [loss_computer1, loss_computer2]
        self.device = device
        self.alpha = alpha

    
    def get_model(self, index):
        return self.loss_computers[index].model
    

    def train(self, batch1, batch2):
        model0 = self.get_model(0)
        model1 = self.get_model(1)
        model0.zero_grad()
        model1.zero_grad()

        loss0 = self.loss_computers[0].compute_loss(batch1)
        loss1 = self.loss_computers[1].compute_loss(batch2)

        loss = self.alpha*loss0 + (1-self.alpha)*loss1
        loss.backward()
        self.optimizer.step()
        return loss.cpu().item()

    def compute(self, batch1, batch2):
        return [lc.compute(batch) for lc, batch in zip(self.loss_computers,[batch1,batch2]) ]
        
    def evaluate(self, batch1, batch2):
        return [lc.evaluate(batch) for lc, batch in zip(self.loss_computers,[batch1,batch2]) ]

    def validate(self, batch1, batch2):
        return [lc.validate(batch) for lc, batch in zip(self.loss_computers,[batch1,batch2]) ]


class EpochTrainerJoint():
    def __init__(self, model1, model2, criterion1, criterion2, optimizer, alpha, device="cpu"):
        loss_computer1 = LossComputer(model1, criterion1, device)
        loss_computer2 = LossComputer(model2, criterion2, device)

        self.trainer = BatchTraninerJoint(loss_computer1, loss_computer2, optimizer, alpha, device)

    def get_model(self, idx):
        return self.trainer.get_model(idx)

    def apply_to_batches(self, function, loader1, loader2):
        return [function(batch1, batch2) for batch1, batch2 in zip(loader1,loader2)]

    def train(self, loader1, loader2):
        return self.apply_to_batches(self.trainer.train, loader1, loader2)

    def validate(self, loader1, loader2):
        self.apply_to_batches(self.trainer.validate, loader1, loader2)

    def evaluate(self, loader1, loader2):
        d1, d2 = zip(*self.apply_to_batches(self.trainer.evaluate, loader1, loader2))
        v1 = [torch.cat(l) for l in zip(*d1)]
        v2 = [torch.cat(l) for l in zip(*d2)]
        return v1,v2


class MetricsComputerIS():
    def __init__(self, ztransformer):
        self.ztransformer = ztransformer

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


class TrainHelperJoint():
    def __init__(
            self, trainer: EpochTrainerJoint,
            loader_tr1, loader_tr2,
            loader_val1, loader_val2,
            loader_ts1, loader_ts2, 
            display_criterion1: Callable[[torch.Tensor, torch.Tensor], float],
            display_criterion2: Callable[[torch.Tensor, torch.Tensor], float]):
        
        self.trainer = trainer
        self.loaders_tr = [loader_tr1, loader_tr2]
        self.loaders_ts = [loader_ts1, loader_ts2]
        self.loaders_val = [loader_val1, loader_val2]
        self.display_criterion1 = display_criterion1
        self.display_criterion2 = display_criterion2


 
    def compute_metric(self, loaders):
        v1, v2 = self.trainer.evaluate(*loaders)
        return (self.display_criterion1(*v1[-2:]), self.display_criterion2(*v2[-2:]))
    
    def get_models(self):
        return [self.trainer.get_model(0), self.trainer.get_model(1)]
    
    def get_state_dicts(self):
        return [copy.deepcopy(m.state_dict()) for m in self.get_models()]

    def load_state_dicts(self, state_dicts):
        return [m.load_state_dict(sd) for m, sd in zip(self.get_models(), state_dicts)]


    def train(self, n_epoch):
        best_val_models  = self.get_state_dicts()# copy.deepcopy(self.trainer.model.state_dict())
        #train_metrics = []
        validation_metrics = [self.compute_metric(self.loaders_val)] 
        test_metric = self.compute_metric(self.loaders_ts) 
    
        for epoch in range(1, n_epoch+1):
            self.trainer.train(*self.loaders_tr)
            loss_val = self.compute_metric(self.loaders_val)
            # loss_tr = self.compute_metric(self.loader_tr)
            # loss_ts = self.compute_metric(self.loader_ts)
            # print('[%d/%d]: loss_train: %.3f loss_val %.3f loss_ts %.3f' % (
                    # (epoch), n_epoch, loss_tr, loss_val, loss_ts))
            

            if loss_val[0] < np.min([v[0] for v in validation_metrics]):
                print("best val epoch:", epoch)
                loss_tr = self.compute_metric(self.loaders_tr)
                loss_ts = self.compute_metric(self.loaders_ts)
                test_metric = loss_ts 
                best_val_models = self.get_state_dicts()# copy.deepcopy(self.trainer.model.state_dict())
                print(f'[{epoch}/{n_epoch}]: loss_train: {loss_tr} loss_val {loss_val} loss_ts {loss_ts}' % (
                    (epoch), n_epoch, loss_tr, loss_val, loss_ts))
            validation_metrics.append(loss_val)

            # train_metrics.append(loss_tr)
            # validation_metrics.append(loss_val)
            # test_metrics.append(loss_ts)
        
        self.load_state_dicts(best_val_models)
        print(f"Final: {test_metric}")
        return test_metric
            
            
