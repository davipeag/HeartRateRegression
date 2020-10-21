

import torch
import copy

import PPG
from PPG import UtilitiesDataXY 

from PPG.Models import SnippetConvolutionalTransformer
from PPG.TrainerXY import (EpochTrainerXY, MetricsComputerXY, TrainHelperXY)
from preprocessing_utils import ZTransformer2


class AttentionFullTrainer():
    def __init__(self, dfs, device, ts_sub, val_sub):
        self.transformers = PPG.AttentionDefaults.get_preprocessing_transformer()    
        self.dfs = dfs
        self.ts_sub = ts_sub
        self.val_sub = val_sub
        self.device = device

    def train(
        self,
        nfeatures=4,
        conv_filters=64,
        nconv_layers = 4,
        conv_dropout=0.5, 
        nenc_layers=2,
        ndec_layers=2,
        nhead=4,
        feedforward_expansion=2,
        nlin_layers = 2,
        lin_size = 32,
        lin_dropout = 0,
        lr = 0.001,
        weight_decay= 0.0001,
        batch_size = 128,
        ts_sub = 0,
        val_sub = 4
        ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) vor v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size")]

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.DataLoaderFactory(
            self.transformers, self.dfs, batch_size_tr=batch_size
            ).make_loaders(ts_sub, val_sub)



        net = PPG.Models.SnippetConvolutionalTransformer(**net_args).to(self.device)
        criterion = torch.nn.MSELoss().to(self.device)# nn.L1Loss().to(args["device"]) #nn.CrossEntropyLoss().to(args["device"])
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                    weight_decay=weight_decay)
        

        epoch_trainer = EpochTrainerXY(net, optimizer, criterion, self.device)
        ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                    'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                    'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])
        metrics_comuter = MetricsComputerXY(ztransformer)

        train_helper = TrainHelperXY(epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)
        
        metric = train_helper.train(30)

        p = metrics_comuter.inverse_transform_label(epoch_trainer.evaluate(loader_ts)[2])

        return {
            "args": args,
            "predictions": p,
            "metric": metric
        }

# #%%
# class ATry():
#     def __init__(self):
#         self.a = 1
#     def atry(self, a=2, b=3):
#         return locals()

# type(ATry().atry())