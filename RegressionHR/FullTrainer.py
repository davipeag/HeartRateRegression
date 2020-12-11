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
from RegressionHR import Preprocessing

import Models
from Models import BaseModels

import sklearn.metrics

# from torch.nn.functional import sigmoid
from torch import sigmoid

accuracy = lambda y,p: (torch.sum((sigmoid(p) > 0.5)== y)/len(p)).detach().cpu().item() 

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
        disc_nlayers=3,
        disc_layer_size=32,
        disc_dropout_rate = 0
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

        sample_step_ratio = ts_per_samples[0]

        transformers2 = RegressionHR.PceLstmDefaults.PamapPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio)
        
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample)

            
            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
                transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
                transformers_val=transformers_val, transformers_ts=transformers_ts, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, val_sub)

            #accuracy = lambda y,p: (torch.sum((sigmoid(p) > 0.5)== y)/len(p)).detach().cpu().item() 

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: (criterion2(p, y).cpu().item(), accuracy(y,p)) # torch.mean(torch.abs(y-p)).detach().cpu().item()
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
        period_s=4,
        disc_nlayers=3,
        disc_layer_size=32,
        disc_dropout_rate = 0
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples", "alpha", "step_s", "period_s")]
        frequency_hz = 100
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(frequency_hz*step_s))-3
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

        sample_step_ratio = ts_per_samples[0]

        transformers2 = RegressionHR.PceLstmDefaults.PamapPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio)
        
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample)

            
            # loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
            #     transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            #     transformers_val=transformers_val, transformers_ts=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            # ).make_loaders(ts_sub, val_sub)

            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
                transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
                dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, 0.8)

            

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: (criterion2(p, y).cpu().item(), accuracy(y,p)) # torch.mean(torch.abs(y-p)).detach().cpu().item()
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



class PceLstmDiscriminatorFullTrainerJointValidation2():
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
        period_s=4,
        ts_per_is = 2,
        is_h_size = 32,
        disc_nlayers=3,
        disc_layer_size=32,
        disc_dropout_rate = 0
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples", "alpha", "step_s", "period_s")]
        frequency_hz = 100

        net_args["sample_per_ts"] = int(frequency_hz*period_s)
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(frequency_hz*step_s))-3
        transformers_ts = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_ts, ts_per_is=ts_per_is, sample_step_ratio=1)

        nets = list(map(lambda n: n.to(self.device), 
                                 RegressionHR.PceLstmModel.parametrized_encoder_make_pce_lstm_and_discriminator(**net_args)))

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

        sample_step_ratio = ts_per_samples[0]

        transformers2 = RegressionHR.PceLstmDefaults.PamapPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio, ts_per_is=ts_per_is)
        
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample, ts_per_is=ts_per_is, sample_step_ratio=1)

            
            # loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
            #     transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            #     transformers_val=transformers_val, transformers_ts=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            # ).make_loaders(ts_sub, val_sub)

            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
                transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
                dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, 0.8)
            
            #accuracy = lambda y,p: (torch.sum((p > 0.5)== y)/len(p)).detach().cpu().item() 

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: (criterion2(p, y).cpu().item(), accuracy(y,p)) # torch.mean(torch.abs(y-p)).detach().cpu().item()
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


    
class DaliaPceLstmDiscriminatorFullTrainerJointValidation2():
    def __init__(self, dfs, device, nepoch = 40):
        self.dfs = dfs
        self.device = device
        self.transformers = RegressionHR.PceLstmDefaults.DaliaPreprocessingTransformerGetter()
        self.nepoch = nepoch
    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        nattrs=7,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=5,
        val_sub=4,
        ts_per_samples = [40],
        alpha = 0.8,
        step_s=2,
        period_s=4,
        ts_per_is = 2,
        is_h_size = 32,
        disc_nlayers=3,
        disc_layer_size=32,
        disc_dropout_rate = 0
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples", "alpha", "step_s", "period_s")]
        frequency_hz = 100

        net_args["sample_per_ts"] = int(frequency_hz*period_s)
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(frequency_hz*step_s))-3
        transformers_ts = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_ts, ts_per_is=ts_per_is)

        nets = list(map(lambda n: n.to(self.device), 
                                 RegressionHR.PceLstmModel.parametrized_encoder_make_pce_lstm_and_discriminator(**net_args)))

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

        sample_step_ratio = ts_per_samples[0]

        transformers2 = RegressionHR.PceLstmDefaults.DaliaPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio, ts_per_is=ts_per_is)

        transformers2_val = RegressionHR.PceLstmDefaults.DaliaPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio//2, ts_per_is=ts_per_is)
         
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, transformers_val=transformers2_val, transformers_ts=transformers2_val, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample, ts_per_is=ts_per_is, sample_step_ratio=1)

            
            # loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
            #     transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            #     transformers_val=transformers_val, transformers_ts=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            # ).make_loaders(ts_sub, val_sub)

            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
                transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
                dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, 0.8)
            
            #accuracy = lambda y,p: (torch.sum((p > 0.5)== y)/len(p)).detach().cpu().item() 

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: (criterion2(p, y).cpu().item(), accuracy(y,p)) # torch.mean(torch.abs(y-p)).detach().cpu().item()
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


class DaliaPceLstmCossineLimilarityFullTrainerJointValidation():
    def __init__(self, dfs, device, nepoch = 40):
        self.dfs = dfs
        self.device = device
        self.transformers = RegressionHR.PceLstmDefaults.DaliaPreprocessingTransformerGetter()
        self.nepoch = nepoch
    def train(
        self,
        ts_h_size = 32,
        lstm_size=32,
        lstm_input = 128,
        dropout_rate = 0,
        nattrs=7,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=5,
        val_sub=4,
        ts_per_samples = [40],
        alpha = 0.8,
        step_s=2,
        period_s=4,
        ts_per_is = 2,
        is_h_size = 32
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples", "alpha", "step_s", "period_s")]
        frequency_hz = 100

        net_args["sample_per_ts"] = int(frequency_hz*period_s)
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(frequency_hz*step_s))-3
        transformers_ts = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_ts, ts_per_is=ts_per_is)

        nets = list(map(lambda n: n.to(self.device), 
                                 RegressionHR.PceLstmModel.parametrized_encoder_make_pce_lstm_and_cossine_similarity(**net_args)))

        pce_lstm, pce_discriminator = nets

        [PPG.Models.initialize_weights(net) for net in nets]

        criterion1 = torch.nn.L1Loss().to(self.device)
        # criterion2 = torch.nn.BCELoss().to(self.device)
        # criterion2 = torch.nn.BCEWithLogitsLoss().to(self.device)
        criterion2 = torch.nn.L1Loss().to(self.device)

        optimizer = torch.optim.Adam([{"params": pce_lstm.parameters()},
                                      #{"params": pce_discriminator.discriminator.parameters()}
                                      ], lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = RegressionHR.TrainerJoint.EpochTrainerJoint(
            pce_lstm, pce_discriminator, criterion1, criterion2, optimizer, alpha, self.device)
        
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = PPG.TrainerIS.MetricsComputerIS(ztransformer)

        sample_step_ratio = ts_per_samples[0]

        transformers2 = RegressionHR.PceLstmDefaults.DaliaPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio, ts_per_is=ts_per_is)

        transformers2_val = RegressionHR.PceLstmDefaults.DaliaPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio//2, ts_per_is=ts_per_is)
         
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, transformers_val=transformers2_val, transformers_ts=transformers2_val, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample, ts_per_is=ts_per_is, sample_step_ratio=1)

            
            # loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
            #     transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            #     transformers_val=transformers_val, transformers_ts=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            # ).make_loaders(ts_sub, val_sub)

            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
                transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
                dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, 0.8)
            
            #accuracy = lambda y,p: (torch.sum((p > 0.5)== y)/len(p)).detach().cpu().item() 

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: (criterion2(p, y).cpu().item()) # torch.mean(torch.abs(y-p)).detach().cpu().item()
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


class PceLstmCosineSimilarityFullTrainerJointValidation():
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
        period_s=4,
        ts_per_is = 2,
        is_h_size = 32
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples", "alpha", "step_s", "period_s")]
        frequency_hz = 100

        net_args["sample_per_ts"] = int(frequency_hz*period_s)
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(frequency_hz*step_s))-3
        transformers_ts = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_ts, ts_per_is=ts_per_is, sample_step_ratio=1)

        nets = list(map(lambda n: n.to(self.device), 
                                 RegressionHR.PceLstmModel.parametrized_encoder_make_pce_lstm_and_cossine_similarity(**net_args)))

        pce_lstm, pce_discriminator = nets

        [PPG.Models.initialize_weights(net) for net in nets]

        criterion1 = torch.nn.L1Loss().to(self.device)
        # criterion2 = torch.nn.BCELoss().to(self.device)
        # criterion2 = torch.nn.BCEWithLogitsLoss().to(self.device)
        criterion2 = torch.nn.L1Loss().to(self.device)

        optimizer = torch.optim.Adam([{"params": pce_lstm.parameters()},
                                      #{"params": pce_discriminator.discriminator.parameters()}
                                      ], lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = RegressionHR.TrainerJoint.EpochTrainerJoint(
            pce_lstm, pce_discriminator, criterion1, criterion2, optimizer, alpha, self.device)
        
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = PPG.TrainerIS.MetricsComputerIS(ztransformer)

        sample_step_ratio = ts_per_samples[0]

        transformers2 = RegressionHR.PceLstmDefaults.PamapPceDecoderPreprocessingTransformerGetter()(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz, sample_step_ratio=sample_step_ratio, ts_per_is=ts_per_is)
        
        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)

        for ts_per_sample in ts_per_samples:

            transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample, ts_per_is=ts_per_is, sample_step_ratio=1)

            
            # loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.DataLoaderFactory(
            #     transformers_tr, dfs= self.dfs, batch_size_tr=batch_size,
            #     transformers_val=transformers_val, transformers_ts=transformers_val, dataset_cls=PPG.UtilitiesDataXY.ISDataset
            # ).make_loaders(ts_sub, val_sub)

            loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
                transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
                dataset_cls=PPG.UtilitiesDataXY.ISDataset
            ).make_loaders(ts_sub, 0.8)
            
            #accuracy = lambda y,p: (torch.sum((p > 0.5)== y)/len(p)).detach().cpu().item() 

            train_helper = RegressionHR.TrainerJoint.TrainHelperJoint(
                epoch_trainer, loader_tr1, loader_tr2, loader_val1, loader_val2,
                loader_ts1, loader_ts2,
                metrics_computer.mae,
                lambda y,p: (criterion2(p, y).cpu().item(), accuracy(y,p)) # torch.mean(torch.abs(y-p)).detach().cpu().item()
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



class IteractiveFFNNFullTrainerJointValidation():
    def __init__(
        self,
        dfs,
        device,
        nepoch = 40,
        net_builder_cls = Models.BaseModels.IterativeSkipFFNN,
        transformer_getter_cls = RegressionHR.Preprocessing.FFNNPreprocessingTransformerGetter,
        dataset_name = "pamap2",
        feature_columns = [
            'heart_rate', 
            'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6'],
        frequency_hz = 100,
        
        ):
        self.dfs = dfs
        self.device = device
        self.transformers = transformer_getter_cls(feature_columns, dataset_name)
        self.nepoch = nepoch
        self.net_builder_cls = net_builder_cls
        self.feature_columns = feature_columns
        self.frequency_hz = frequency_hz
        
    def train(
        self,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=128,
        ts_sub=5,
        ts_per_sample = 40,
        step_s=2,
        period_s=4,
        **net_args
    ):
        args = locals()
        args.pop("self")
        net_args["input_features"] = len(self.feature_columns)
        frequency_hz = self.frequency_hz
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(frequency_hz*step_s))-3
        transformers_ts = self.transformers(period_s = period_s, step_s = step_s, frequency_hz = frequency_hz, ts_per_sample=ts_per_sample_ts)

        net = self.net_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(net)

        criterion = torch.nn.L1Loss().to(self.device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = PPG.TrainerIS.EpochTrainerIS(
            model = net, criterion = criterion, optimizer = optimizer, device = self.device
        )  
        
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = PPG.TrainerIS.MetricsComputerIS(ztransformer)
        
        transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz, ts_per_sample=ts_per_sample)

    
        loader_tr, loader_val, loader_ts = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
            dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)


        train_helper = PPG.TrainerIS.TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_computer.mae
        )         
            
        metric = train_helper.train(self.nepoch)

        p = [metrics_computer.inverse_transform_label(v)
             for v in  epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }


class SingleNetFullTrainerJointValidationIS():
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        net_builder_cls,
        transformer_getter_cls,
        dataset_name,
        feature_columns,
        frequency_hz,
        input_features_parameter_name,
        additional_args = dict(),
        additional_net_args = dict()
        ):
        self.dfs = dfs
        self.device = device
        self.transformers = transformer_getter_cls(feature_columns, dataset_name)
        self.nepoch = nepoch
        self.net_builder_cls = net_builder_cls
        self.feature_columns = feature_columns
        self.frequency_hz = frequency_hz
        self.input_features_parameter_name = input_features_parameter_name
        self.additional_args = additional_args
        self.additional_net_args = additional_net_args
    def train(
        self,
        lr,
        weight_decay,
        batch_size,
        ts_sub,
        ts_per_sample,
        ts_per_is,
        step_s,
        period_s,
        **net_args
    ):
        net_args[self.input_features_parameter_name] = len(self.feature_columns)
        net_args = {**net_args, **self.additional_net_args}
        args = locals()
        args = {**args, **self.additional_args}
        args.pop("self")
        frequency_hz = self.frequency_hz
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(frequency_hz*step_s))-3
        transformers_ts = self.transformers(
            period_s = period_s, step_s = step_s, frequency_hz = frequency_hz,
            ts_per_sample=ts_per_sample_ts, ts_per_is=ts_per_is)

        net = self.net_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(net)

        criterion = torch.nn.L1Loss().to(self.device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        epoch_trainer = PPG.TrainerIS.EpochTrainerIS(
            model = net, criterion = criterion, optimizer = optimizer, device = self.device
        )  
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = PPG.TrainerIS.MetricsComputerIS(ztransformer)
        
        transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz,
                                            ts_per_sample=ts_per_sample, ts_per_is=ts_per_is)

    
        loader_tr, loader_val, loader_ts = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
            dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)


        train_helper = PPG.TrainerIS.TrainHelperIS(
            epoch_trainer, loader_tr, loader_val, loader_ts, metrics_computer.mae
        )         
            
        metric = train_helper.train(self.nepoch)

        p = [metrics_computer.inverse_transform_label(v)
             for v in  epoch_trainer.evaluate(loader_ts)[-2:]]

        return {
            "args": args,
            "predictions": p,
            "metric": metric,
            "run_class": self.__class__.__name__
        }

class NoPceLstmPamap2FullTrainerJointValidation(SingleNetFullTrainerJointValidationIS):
    def __init__(self, dfs, device, nepoch, feature_columns=[
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
            'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
            'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
            'a_ymag', 'a_zmag'
        ]):
        super(NoPceLstmPamap2FullTrainerJointValidation, self).__init__(
            dfs, device, nepoch, RegressionHR.PceLstmModel.make_par_enc_no_pce_lstm,
            RegressionHR.Preprocessing.PceLstmTransformerGetter, "pamap2", feature_columns,
             100, "nattrs")
 
class NoPceLstmDaliaFullTrainerJointValidation(SingleNetFullTrainerJointValidationIS):
    def __init__(self, dfs, device, nepoch, feature_columns=[
            'heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
            'chest-ACC-0','chest-ACC-1', 'chest-ACC-2'
        ]):
        super(NoPceLstmDaliaFullTrainerJointValidation, self).__init__(
            dfs, device, nepoch, RegressionHR.PceLstmModel.make_par_enc_no_pce_lstm,
            RegressionHR.Preprocessing.PceLstmTransformerGetter, "dalia", feature_columns,
             32, "nattrs")

class IteractiveFFNNDaliaFullTrainerJointValidation(SingleNetFullTrainerJointValidationIS):
        def __init__(self, dfs, device, nepoch, feature_columns=[
                'heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                #'chest-ACC-0','chest-ACC-1', 'chest-ACC-2'
            ]):
            super(IteractiveFFNNDaliaFullTrainerJointValidation, self).__init__(
                dfs, device, nepoch, Models.BaseModels.IterativeSkipFFNN,
                RegressionHR.Preprocessing.FFNNPreprocessingTransformerGetter, "dalia", feature_columns,
                32, "input_features")
