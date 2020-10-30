

import torch
import copy

import PPG
from PPG import UtilitiesDataXY

from PPG.Models import (SnippetConvolutionalTransformer, initialize_weights)
from PPG.TrainerXY import (EpochTrainerXY, MetricsComputerXY, TrainHelperXY)
from PPG.TrainerIS import (EpochTrainerIS, MetricsComputerIS, TrainHelperIS)
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
        nconv_layers=4,
        conv_dropout=0.5,
        nenc_layers=2,
        ndec_layers=2,
        nhead=4,
        feedforward_expansion=2,
        nlin_layers=2,
        lin_size=32,
        lin_dropout=0,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size")]

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.DataLoaderFactory(
            self.transformers, self.dfs, batch_size_tr=batch_size
        ).make_loaders(ts_sub, val_sub)

        net = PPG.Models.SnippetConvolutionalTransformer(
            **net_args).to(self.device)
        # nn.L1Loss().to(args["device"]) #nn.CrossEntropyLoss().to(args["device"])
        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerXY(net, optimizer, criterion, self.device)
        ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                                      'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                                      'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])
        metrics_comuter = MetricsComputerXY(ztransformer)

        train_helper = TrainHelperXY(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(30)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric
        }

class JointValAttentionFullTrainer():
    def __init__(self, dfs, device):
        self.transformers = PPG.AttentionDefaults.get_preprocessing_transformer()
        self.dfs = dfs
        self.device = device

    def train(
        self,
        nfeatures=4,
        conv_filters=64,
        nconv_layers=4,
        conv_dropout=0.5,
        nenc_layers=2,
        ndec_layers=2,
        nhead=4,
        feedforward_expansion=2,
        nlin_layers=2,
        lin_size=32,
        lin_dropout=0,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size")]

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.JointTrValDataLoaderFactory(
            self.transformers, dfs = self.dfs, batch_size_tr=batch_size
        ).make_loaders(ts_sub, 0.8)

        net = PPG.Models.SnippetConvolutionalTransformer(
            **net_args).to(self.device)
        # nn.L1Loss().to(args["device"]) #nn.CrossEntropyLoss().to(args["device"])
        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerXY(net, optimizer, criterion, self.device)
        ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                                      'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                                      'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])
        metrics_comuter = MetricsComputerXY(ztransformer)

        train_helper = TrainHelperXY(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(30)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric
        }


class PceLstmFullTrainer():
    def __init__(self, dfs, device, ts_sub, val_sub):
        self.dfs = dfs
        self.ts_sub = ts_sub
        self.val_sub = val_sub
        self.device = device

    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        bvp_count=12,
        nattrs=5,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size")]

        transformers_tr = PPG.PceLstmDefaults.get_preprocessing_transformer()
        ts_per_sample = int(len(self.dfs[val_sub])/(32*8))-3
        transformers_val = PPG.PceLstmDefaults.get_preprocessing_transformer(ts_per_sample=ts_per_sample)

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.DataLoaderFactory(
            transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            transformers_val=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, val_sub)

        net = PPG.PceLstmModel.make_pce_lstm(
            **net_args).to(self.device)
        initialize_weights(net)
        criterion = torch.nn.L1Loss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerIS(net, optimizer, criterion, self.device)
        ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                                      'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                                      'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])
        metrics_comuter = MetricsComputerIS(ztransformer)

        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(30)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }


class NoHrPceLstmFullTrainer():
    def __init__(self, dfs, device):
        self.dfs = dfs
        self.device = device
        self.train_helper = None
        self.metrics_computer = None

    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        bvp_count=12,
        nattrs=5,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size")]

        transformers_tr = PPG.PceLstmDefaults.get_preprocessing_transformer()
        ts_per_sample = int(len(self.dfs[ts_sub])/(32*8))-3
        transformers_ts = PPG.PceLstmDefaults.get_preprocessing_transformer(ts_per_sample=ts_per_sample)

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.DataLoaderFactory(
            transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            transformers_ts=transformers_ts, dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, val_sub)

        net = PPG.NoHrPceLstmModel.make_no_hr_pce_lstm(
            **net_args).to(self.device)
        initialize_weights(net)
    
        criterion = torch.nn.L1Loss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerIS(net, optimizer, criterion, self.device)
        ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                                      'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                                      'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])
        metrics_comuter = MetricsComputerIS(ztransformer)
        self.metrics_computer = metrics_comuter
        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)
        self.train_helper = train_helper
        metric = train_helper.train(30)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }



class JointValNoHrPceLstmFullTrainer():
    def __init__(self, dfs, device):
        self.dfs = dfs
        self.device = device

    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        bvp_count=12,
        nattrs=5,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size")]

        transformers_tr = PPG.PceLstmDefaults.get_preprocessing_transformer()
        ts_per_sample = int(len(self.dfs[ts_sub])/(32*8))-3
        transformers_ts = PPG.PceLstmDefaults.get_preprocessing_transformer(ts_per_sample=ts_per_sample)

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            transformers_ts=transformers_ts, dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)

        net = PPG.NoHrPceLstmModel.make_no_hr_pce_lstm(
            **net_args).to(self.device)
        initialize_weights(net)
    
        criterion = torch.nn.L1Loss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerIS(net, optimizer, criterion, self.device)
        ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                                      'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                                      'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])
        metrics_comuter = MetricsComputerIS(ztransformer)

        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(40)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }


class IeeeJointValNoHrPceLstmFullTrainer():
    def __init__(self, dfs, device):
        self.dfs = dfs
        self.device = device

    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        bvp_count=12,
        nattrs=5,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size")]

        transformers_tr = PPG.PceLstmDefaults.get_preprocessing_transformer_ieee()
        ts_per_sample = int(len(self.dfs[ts_sub])/(32*8))-3
        transformers_ts = PPG.PceLstmDefaults.get_preprocessing_transformer_ieee(ts_per_sample=ts_per_sample)

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            transformers_ts=transformers_ts, dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)

        net = PPG.IeeeNoHrPceLstmModel.make_no_hr_pce_lstm(
            **net_args).to(self.device)
        initialize_weights(net)
    
        criterion = torch.nn.L1Loss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerIS(net, optimizer, criterion, self.device)
        ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                                      'wrist-BVP-0', 'wrist-BVP-1'])
        metrics_comuter = MetricsComputerIS(ztransformer)

        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(200)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }