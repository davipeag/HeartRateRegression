import torch
import copy

import PPG
import RegressionHR
from PPG import UtilitiesDataXY

from PPG.Models import (SnippetConvolutionalTransformer, initialize_weights)
from PPG.TrainerXY import (EpochTrainerXY, MetricsComputerXY, TrainHelperXY)
from PPG.TrainerIS import (EpochTrainerIS, MetricsComputerIS, TrainHelperIS)
from preprocessing_utils import ZTransformer2


from RegressionHR import FullTrainer
from RegressionHR import PceLstmDefaults
from RegressionHR import PceLstmModel
from RegressionHR import TrainerJoint
from RegressionHR import  UtilitiesData


class PceLstmFullTrainer():
    def __init__(self, dfs, device, ts_sub, val_sub, nepoch = 40):
        self.dfs = dfs
        self.ts_sub = ts_sub
        self.val_sub = val_sub
        self.device = device
        self.transformers = RegressionHR.PceLstmDefaults.PamapPreprocessingTransformerGetter()
        self.nepoch = nepoch
    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        nattrs=40,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=5,
        val_sub=4,
        ts_per_samples = [30, 60],
        period_s = 4,
        step_s = 2
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size",
                "ts_per_samples", "period_s", "step_s")]
        frequency_hz = 100
        
        ldf = min(len(self.dfs[self.val_sub]), len(self.dfs[self.ts_sub]))
        ts_per_sample_val = int(ldf/(frequency_hz*step_s))-3
        transformers_val = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_val)

        net = RegressionHR.PceLstmModel.make_pce_lstm(
            **net_args).to(self.device)
        initialize_weights(net)
        criterion = torch.nn.L1Loss().to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = EpochTrainerIS(net, optimizer, criterion, self.device)
        ztransformer = self.transformers.ztransformer
        metrics_comuter = MetricsComputerIS(ztransformer)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample)

            loader_tr, loader_val, loader_ts = UtilitiesDataXY.DataLoaderFactory(
                transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
                transformers_val=transformers_val, transformers_ts=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, val_sub)

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



class PceLstmDiscriminatorFullTrainer():
    def __init__(self, dfs, device, nepoch = 40):
        self.dfs = dfs
        self.device = device
        self.transformers = RegressionHR.PceLstmDefaults.PamapPreprocessingTransformerGetter()
        self.nepoch = nepoch
    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        nattrs=40,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=5,
        val_sub=4,
        ts_per_samples = [40],
        alpha = 0.8,
        period_s = 4,
        step_s = 2,
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in (
            "ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples", "alpha",
            "period_s", "step_s")]
        frequency_hz = 100
        
        
        ts_per_sample_val = int(len(self.dfs[val_sub])/(frequency_hz*step_s))-3
        transformers_val = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_val)

        ts_per_sample_ts = int(len(self.dfs[ts_sub])/(frequency_hz*step_s))-3
        transformers_ts = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_ts)

        nets = list(map(lambda n: n.to(self.device), 
                                 RegressionHR.PceLstmModel.make_pce_lstm_and_discriminator(**net_args)))

        pce_lstm, pce_discriminator = nets

        [PPG.Models.initialize_weights(net) for net in nets]

        criterion1 = torch.nn.L1Loss().to(self.device)
        # criterion2 = torch.nn.BCELoss().to(self.device)
        criterion2 = torch.nn.BCEWithLogitsLoss().to(self.device)

        optimizer = torch.optim.Adam([{"params": pce_lstm.parameters()},
                                      {"params": pce_discriminator.discriminator.parameters()}], lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = RegressionHR.TrainerJoint.EpochTrainerJoint(
            pce_lstm, pce_discriminator, criterion1, criterion2, optimizer, alpha, self.device)
        
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = PPG.TrainerIS.MetricsComputerIS(ztransformer)

        transformers2 = RegressionHR.PceLstmDefaults.PamapPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=10)
        
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample)

            
            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
                transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
                transformers_val=transformers_val, transformers_ts=transformers_ts, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, val_sub)

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: torch.mean(torch.abs(y-p)).detach().cpu().item()
            )
            
            print("about to train:")
            metric = train_helper.train(self.nepoch)

        d0, d1 = epoch_trainer.evaluate(loader_ts1, loader_ts2) 

        p = [metrics_computer.inverse_transform_label(v)
             for v in d0[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }


class PceLstmDiscriminatorFullTrainerJointValidation():
    def __init__(self, dfs, device, nepoch = 40):
        self.dfs = dfs
        self.device = device
        self.transformers = RegressionHR.PceLstmDefaults.PamapPreprocessingTransformerGetter()
        self.nepoch = nepoch
    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        nattrs=40,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=5,
        val_sub=4,
        ts_per_samples = [40],
        alpha = 0.8,
        step_s=2,
        period_s=4

    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples", "alpha", "step_s", "period_s")]
        frequency_hz = 100
        
        ldf = min(len(self.dfs[val_sub]), len(self.dfs[ts_sub]))
        ts_per_sample_val = int(ldf/(frequency_hz*step_s))-3
        transformers_val = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_val)

        nets = list(map(lambda n: n.to(self.device), 
                                 RegressionHR.PceLstmModel.make_pce_lstm_and_discriminator(**net_args)))

        pce_lstm, pce_discriminator = nets

        [PPG.Models.initialize_weights(net) for net in nets]

        criterion1 = torch.nn.L1Loss().to(self.device)
        # criterion2 = torch.nn.BCELoss().to(self.device)
        criterion2 = torch.nn.BCEWithLogitsLoss().to(self.device)

        optimizer = torch.optim.Adam([{"params": pce_lstm.parameters()},
                                      {"params": pce_discriminator.discriminator.parameters()}], lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = RegressionHR.TrainerJoint.EpochTrainerJoint(
            pce_lstm, pce_discriminator, criterion1, criterion2, optimizer, alpha, self.device)
        
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = PPG.TrainerIS.MetricsComputerIS(ztransformer)

        transformers2 = RegressionHR.PceLstmDefaults.PamapPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=10)
        
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample)

            
            # loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
            #     transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            #     transformers_val=transformers_val, transformers_ts=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            # ).make_loaders(ts_sub, val_sub)

            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
                transformers_tr, transformers_ts=transformers_val, dfs = self.dfs, batch_size_tr=batch_size,
                dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, 0.8)

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: torch.mean(torch.abs(y-p)).detach().cpu().item()
            )
            
            print("about to train:")
            metric = train_helper.train(self.nepoch)

        d0, d1 = epoch_trainer.evaluate(loader_ts1, loader_ts2) 

        p = [metrics_computer.inverse_transform_label(v)
             for v in d0[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }