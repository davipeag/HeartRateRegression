

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
        self.transformers = PPG.AttentionDefaults.PreprocessingTransformerGetter()
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
            self.transformers(), self.dfs, batch_size_tr=batch_size
        ).make_loaders(ts_sub, val_sub)

        net = PPG.Models.SnippetConvolutionalTransformer(
            **net_args).to(self.device)
        # nn.L1Loss().to(args["device"]) #nn.CrossEntropyLoss().to(args["device"])
        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerXY(net, optimizer, criterion, self.device)
        ztransformer = self.transformers.ztransformer
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
    def __init__(self, dfs, device, nepoch = 30, criterion = torch.nn.MSELoss()):
        self.transformers = PPG.AttentionDefaults.PreprocessingTransformerGetter()
        self.dfs = dfs
        self.device = device
        self.nepoch = nepoch
        self.criterion = criterion
        self.net = None
        self.transforer_lr_multiplier = 1
        self.cnn_multiplier = 1
        self.regressor_multiplier = 1


    def make_net(self, net_args, force=False):
        if ((self.net is None) or force):
            self.net = PPG.Models.SnippetConvolutionalTransformer(
            **net_args).to(self.device)
        return self.net
    
    def make_optimizer(self, net, lr, weight_decay):
        
        return torch.optim.Adam(
            [
                {'params': net.transformer.parameters(), 'lr': self.transforer_lr_multiplier* lr},
                {'params': net.conv_net.parameters(), 'lr': self.cnn_multiplier*lr},
                {'params': net.regressor.parameters(), 'lr': self.regressor_multiplier*lr}
            ], lr=lr,
            weight_decay=weight_decay)

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
            self.transformers(), dfs = self.dfs, batch_size_tr=batch_size
        ).make_loaders(ts_sub, 0.8)

        net = self.make_net(net_args)
        # nn.L1Loss().to(args["device"]) #nn.CrossEntropyLoss().to(args["device"])
        #criterion = torch.nn.L1Loss().to(self.device)
        criterion = self.criterion.to(self.device)
        optimizer =  self.make_optimizer(net, lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerXY(net, optimizer, criterion, self.device)
        ztransformer = self.transformers.ztransformer
        metrics_comuter = MetricsComputerXY(ztransformer)

        train_helper = TrainHelperXY(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(self.nepoch)

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
        self.transformers = PPG.PceLstmDefaults.PreprocessingTransformerGetter()

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

        transformers_tr = self.transformers()
        ts_per_sample = int(len(self.dfs[val_sub])/(32*8))-3
        transformers_val = self.transformers(ts_per_sample=ts_per_sample)

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
        ztransformer = self.transformers.ztransformer
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
        self.transformers = PPG.PceLstmDefaults.PreprocessingTransformerGetter()        

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

        transformers_tr = self.transformers()
        ts_per_sample = int(len(self.dfs[ts_sub])/(32*8))-3
        transformers_ts = self.transformers(ts_per_sample=ts_per_sample)

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
        
        ztransformer = self.transformers.ztransformer

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
    def __init__(self, dfs, device, nepoch = 40):
        self.dfs = dfs
        self.device = device
        self.transformers = PPG.PceLstmDefaults.PreprocessingTransformerGetter()
        self.nepoch = nepoch

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

        transformers_tr =self.transformers()
        ts_per_sample = int(len(self.dfs[ts_sub])/(32*8))-3-3
        transformers_ts = self.transformers(ts_per_sample=ts_per_sample)

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
        ztransformer = self.transformers.ztransformer
        metrics_comuter = MetricsComputerIS(ztransformer)

        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(self.nepoch)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }


class IeeeJointValNoHrPceLstmFullTrainer():
    def __init__(self, dfs, device, nrun = 40):
        self.dfs = dfs
        self.device = device
        self.transformers = PPG.PceLstmDefaults.IeeePreprocessingTransformerGetter()
        self.nrun = nrun
    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        bvp_count=12,
        nattrs=5,
        ts_per_sample = 30,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_sample")]

        transformers_tr = self.transformers(ts_per_sample=ts_per_sample)
        step_s = 2
        frequency_hz = 125
        ts_per_sample = int(len(self.dfs[ts_sub])/(frequency_hz*step_s))-3 - 3
        transformers_ts = self.transformers(ts_per_sample=ts_per_sample)

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
        ztransformer = self.transformers.ztransformer
        metrics_comuter = MetricsComputerIS(ztransformer)

        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(self.nrun)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }



class IeeeJointValAttentionFullTrainer():
    def __init__(self, dfs, device):
        self.transformers = PPG.AttentionDefaults.IeeePreprocessingTransformerGetter()
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
        val_sub=4,
        nepoch = 100,
        downsampling_ratio = 32/125
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in (
            "ts_sub", "val_sub", "lr", "weight_decay", "batch_size",
            "nepoch", "downsampling_ratio")]
        self.transformers.set_downsampling_ratio(downsampling_ratio)
        loader_tr, loader_val, loader_ts = UtilitiesDataXY.JointTrValDataLoaderFactory(
            self.transformers(), dfs = self.dfs, batch_size_tr=batch_size
        ).make_loaders(ts_sub, 0.8)

        net = PPG.Models.SnippetConvolutionalTransformer(
            **net_args).to(self.device)
        criterion = torch.nn.L1Loss().to(self.device) 
        #criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerXY(net, optimizer, criterion, self.device)
        ztransformer = self.transformers.ztransformer

        metrics_comuter = MetricsComputerXY(ztransformer)

        train_helper = TrainHelperXY(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(nepoch)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric
        }

class IeeeJointValConvTransfRnnFullTrainer():
    def __init__(self, dfs, device, nrun = 40):
        self.dfs = dfs
        self.device = device
        self.transformers = PPG.PceLstmDefaults.IeeePreprocessingTransformerGetter(donwsampling_ratio=1, do_fft=False)
        self.nrun = nrun
    def train(
        self,

        input_channels,
        bvp_idx = [4,5],
        nfilters=64, dropout_rate=0.1, embedding_size=128,
        bvp_embedding_size=12, predictor_hidden_size=32, feedforward_expansion=2,
        num_encoder_layers= 2, num_decoder_layers=2,
        nheads=4,
        
        ts_per_sample = 30,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=0,
        val_sub=4
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_sample")]

        transformers_tr = self.transformers(ts_per_sample=ts_per_sample)
        step_s = 2
        frequency_hz = 125
        ts_per_sample = int(len(self.dfs[ts_sub])/(frequency_hz*step_s))-3 - 3
        transformers_ts = self.transformers(ts_per_sample=ts_per_sample)

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            transformers_ts=transformers_ts, dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)

        net = PPG.Models.ConvTransfRNN(
            **net_args).to(self.device)
        initialize_weights(net)
    
        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerIS(net, optimizer, criterion, self.device)
        ztransformer = self.transformers.ztransformer
        metrics_comuter = MetricsComputerIS(ztransformer)

        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)

        metric = train_helper.train(self.nrun)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }

class NoHrConvTransfRnnFullTrainer():
    def __init__(self, dfs, device, nepoc=40):
        self.dfs = dfs
        self.device = device
        self.train_helper = None
        self.metrics_computer = None
        self.transformers = PPG.PceLstmDefaults.PreprocessingTransformerGetter(use_fft=True)        
        self.nepoch = nepoc
    def train(
        self,
        input_channels,
        bvp_idx = [4],
        nfilters=64, dropout_rate=0.1, embedding_size=128,
        bvp_embedding_size=12, predictor_hidden_size=32, feedforward_expansion=2,
        num_encoder_layers= 2, num_decoder_layers=2,
        nheads=4,

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

        transformers_tr = self.transformers()
        ts_per_sample = int(len(self.dfs[ts_sub])/(32*8))-3
        transformers_ts = self.transformers(ts_per_sample=ts_per_sample)

        loader_tr, loader_val, loader_ts = UtilitiesDataXY.DataLoaderFactory(
            transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            transformers_ts=transformers_ts, dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, val_sub)

        net = PPG.Models.ConvTransfRNN(
            **net_args).to(self.device)
        initialize_weights(net)
    
        criterion = torch.nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerIS(net, optimizer, criterion, self.device)
        
        ztransformer = self.transformers.ztransformer

        metrics_comuter = MetricsComputerIS(ztransformer)
        self.metrics_computer = metrics_comuter
        train_helper = TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_comuter.mae)
        self.train_helper = train_helper
        metric = train_helper.train(self.nepoch)

        p = [metrics_comuter.inverse_transform_label(v)
             for v in epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }


