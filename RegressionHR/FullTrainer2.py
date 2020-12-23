import torch
import numpy as np
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

import Trainer
from Trainer import BatchTrainers
from Trainer import BatchComputers
from Trainer import ToolBox
from Trainer import DisplayCriterion

from Trainer.BatchComputers import BatchComputerIS, BatchComputerTripletLoss
from Trainer.BatchTrainers import SequentialTrainer

import Models
from Models import BaseModels

import CustomLoss

import sklearn.metrics

# from torch.nn.functional import sigmoid
from torch import sigmoid


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
        additional_net_args = dict(),
        model_name = None
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
        self.model_name = None
    
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

        batch_computer = BatchComputerIS(net, criterion, self.device, self.model_name)
        batch_trainer = SequentialTrainer([batch_computer], optimizer)

        

        # epoch_trainer = PPG.TrainerIS.EpochTrainerIS(
        #     model = net, criterion = criterion, optimizer = optimizer, device = self.device
        # )  
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = RegressionHR.TrainerJoint.MetricsComputerIS(ztransformer)
        
        transformers_tr = self.transformers(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz,
                                            ts_per_sample=ts_per_sample, ts_per_is=ts_per_is)

    
        loader_tr, loader_val, loader_ts = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
            dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)

        epoch_trainer = ToolBox.MultiModelEpochTrainer(batch_trainer)
        
        train_helper = ToolBox.MultiModelTrainHelper(
            epoch_trainer, [loader_tr], [loader_val], [loader_ts],
            [lambda o: metrics_computer.mae(o.label, o.prediction)]
        )
            
        outputs = train_helper.train(self.nepoch)
        outputs = list(outputs.values())[0]
        return {
            **{
            "args": args,
            "run_class": self.__class__.__name__
            }, **outputs
        }


class PceLstmTripletDiscriminator:
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        lstm_builder_cls,
        feature_columns,
        dataset_name,
        frequency_hz,
        input_features_parameter_name,
        additional_args = dict(),
        additional_net_args = dict()
        ):
        self.dfs = dfs
        self.device = device
        self.transformers_decoder = Preprocessing.PceLstmDecoderTripletLossTransformerGetter(feature_columns, dataset_name, frequency_hz)
        self.transformers_lstm = Preprocessing.PceLstmTransformerGetter(feature_columns, dataset_name)
        
        self.nepoch = nepoch
        self.lstm_builder_cls = lstm_builder_cls
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
        val_sub,
        ts_per_sample,
        ts_per_is,
        step_s,
        period_s,
        alpha,
        margin,
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
        transformers_ts_lstm = self.transformers_lstm(
            period_s = period_s, step_s = step_s, frequency_hz = frequency_hz,
            ts_per_sample=ts_per_sample_ts, ts_per_is=ts_per_is)
        
        
        transformers_discr = self.transformers_decoder(period_s = period_s, step_s = step_s, ts_per_is=ts_per_is, sample_step_ratio = ts_per_sample/ts_per_is)



        lstm = self.lstm_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(lstm)
        criterion = torch.nn.L1Loss().to(self.device)
        optimizer = torch.optim.Adam(lstm.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        
        
        batch_computer_lstm = BatchComputerIS(lstm, criterion, self.device, "PceLstm")
        
        discriminator = PceLstmModel.PceEncoderAssembler(ts_encoder = lstm.ts_encoder, is_encoder = lstm.is_encoder)
        criterion_discriminator = CustomLoss.CosineSimilarityTripletLoss(margin=margin)


        batch_computer_discriminator = BatchComputerTripletLoss(discriminator, self.device, criterion_discriminator, "PceDiscriminator")
        
        batch_trainer = SequentialTrainer([batch_computer_lstm, batch_computer_discriminator], optimizer, [alpha, 1-alpha]) # ([batch_computer_lstm, ], optimizer)

        epoch_trainer = ToolBox.MultiModelEpochTrainer(batch_trainer)

        
        ztransformer_lstm = self.transformers_lstm.ztransformer

        metrics_computer_lstm = RegressionHR.TrainerJoint.MetricsComputerIS(ztransformer_lstm)
        metrics_computer_discriminator = DisplayCriterion.CosineSimilarityTripletLoss(margin=margin)
        
        transformers_tr_lstm = self.transformers_lstm(period_s=period_s, step_s=step_s, frequency_hz=frequency_hz,
                                            ts_per_sample=ts_per_sample, ts_per_is=ts_per_is)

    
        loader_tr_lstm, loader_val_lstm, loader_ts_lstm = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr_lstm, transformers_ts=transformers_ts_lstm, dfs = self.dfs, batch_size_tr=batch_size,
            dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)

        loader_tr_discr, loader_val_discr, loader_ts_discr = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            dfs = self.dfs, transformers = transformers_discr, batch_size_tr = batch_size, dataset_cls = RegressionHR.UtilitiesData.TripletPceDiscriminatorDataset
        ).make_loaders(ts_sub, val_sub)

        
        
        loaders_tr = [loader_tr_lstm, loader_tr_discr]
        loaders_val = [loader_val_lstm, loader_val_discr]
        loaders_ts = [loader_ts_lstm, loader_ts_discr]

        train_helper = ToolBox.MultiModelTrainHelper(
            epoch_trainer,loaders_tr, loaders_val, loaders_ts,
            [lambda o: metrics_computer_lstm.mae(o.label, o.prediction), metrics_computer_discriminator]
        )
            
        outputs = train_helper.train(self.nepoch)
        outputs["PceLstm"]["labels"] = metrics_computer_lstm.inverse_transform_label(outputs["PceLstm"]['labels'])
        outputs["PceLstm"]["predictions"] = metrics_computer_lstm.inverse_transform_label(outputs["PceLstm"]['predictions'])
        return {
            **{
            "args": args,
            "run_class": self.__class__.__name__
            }, **outputs
        }


 
class PceLstmBatchComputerMaker:
    def __init__(
        self, lstm_builder_cls, device, input_features_parameter_name, feature_count,
        additional_net_args = dict(),
        criterion = torch.nn.L1Loss(),
        name = "PceLstm"
        ):
        self.device = device
        self.lstm_builder_cls = lstm_builder_cls
        self.input_features_parameter_name = input_features_parameter_name
        self.additional_net_args = additional_net_args
        self.criterion = criterion
        self.feature_count = feature_count
        self.name = name
    
    def make(self, sample_per_ts, ts_per_is, **net_args):
        net_args[self.input_features_parameter_name] = self.feature_count
        net_args["sample_per_ts"] = sample_per_ts
        net_args["ts_per_is"] = ts_per_is
        net_args = {**net_args, **self.additional_net_args}
        
        
        lstm = self.lstm_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(lstm)        
        
        return BatchComputerIS(lstm, self.criterion, self.device, self.name)
        

class TripletDiscriminatorBatchComputerMaker:
    def __init__(self, device, name = "PceDiscriminator"):
        self.device = device
        self.name = name
    
    def make(self, lstm_model, margin):
        discriminator = PceLstmModel.PceEncoderAssembler(ts_encoder = lstm_model.ts_encoder, is_encoder = lstm_model.is_encoder)
        criterion_discriminator = CustomLoss.CosineSimilarityTripletLoss(margin=margin)

        return BatchComputerTripletLoss(discriminator, self.device, criterion_discriminator, self.name)
        

class PceLstmLoadersMaker():
    def __init__(
            self,
            dfs,
            feature_columns,
            dataset_name,
            frequency_hz,
            same_hr = False
        ):
        self.dfs = dfs
        self.transformers_lstm = Preprocessing.PceLstmTransformerGetter(feature_columns, dataset_name, same_hr=same_hr)
        self.ztransformer = self.transformers_lstm.ztransformer
        self.frequency_hz = frequency_hz
    
    def make(self, batch_size, ts_sub, ts_per_sample, ts_per_is, step_s, period_s):
        
        ldf = len(self.dfs[ts_sub])
        ts_per_sample_ts = int(ldf/(self.frequency_hz*step_s))-3
        transformers_ts_lstm = self.transformers_lstm(
            period_s = period_s, step_s = step_s, frequency_hz = self.frequency_hz,
            ts_per_sample=ts_per_sample_ts, ts_per_is=ts_per_is)
                
        transformers_tr_lstm = self.transformers_lstm(period_s=period_s, step_s=step_s, frequency_hz=self.frequency_hz,
                                            ts_per_sample=ts_per_sample, ts_per_is=ts_per_is)

        loader_tr_lstm, loader_val_lstm, loader_ts_lstm = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr_lstm, transformers_ts=transformers_ts_lstm, dfs = self.dfs, batch_size_tr=batch_size,
            dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)

        return loader_tr_lstm, loader_val_lstm, loader_ts_lstm

        
class PceDiscriminatorTripletLossLoadersMaker():
    def __init__(
            self,
            dfs,
            feature_columns,
            dataset_name,
            frequency_hz,
            same_hr = False
        ):
        self.dfs = dfs
        self.transformers_decoder = Preprocessing.PceLstmDecoderTripletLossTransformerGetter(feature_columns, dataset_name, frequency_hz, same_hr=same_hr)
        self.ztransformer = self.transformers_decoder.ztransformer
        self.frequency_hz = frequency_hz
    
    def make(self, batch_size, ts_sub, val_sub, ts_per_sample, ts_per_is, step_s, period_s):
        
        transformers_discr = self.transformers_decoder(period_s = period_s, step_s = step_s, ts_per_is=ts_per_is, sample_step_ratio = ts_per_sample/ts_per_is)

        loader_tr_discr, loader_val_discr, loader_ts_discr = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            dfs = self.dfs, transformers = transformers_discr, batch_size_tr = batch_size, dataset_cls = RegressionHR.UtilitiesData.TripletPceDiscriminatorDataset
        ).make_loaders(ts_sub, val_sub)
        
        return loader_tr_discr, loader_val_discr, loader_ts_discr


class DoubleDatasetPceLstmTripletDiscriminatorFullTrainer:
    def __init__(
        self,
        dfs_ds1,
        dfs_ds2,
        dataset1_name,
        dataset1_feature_columns,
        dataset2_name,
        dataset2_feature_columns,
        dataset1_frequency_hz,
        dataset2_frequency_hz,
        device,
        nepoch,
        input_features_parameter_name,
        ts_sub1,
        ts_sub2,
        val_sub1,
        val_sub2,
        main_index,
        lstm_builder_cls = RegressionHR.PceLstmModel.make_par_enc_pce_lstm,
        additional_net_args1 = dict(),
        additional_net_args2 = dict(),
        ):
        self.dfs1 = dfs_ds1
        self.dfs2 = dfs_ds2

        self.lstm_loader_maker1 = PceLstmLoadersMaker(dfs_ds1, dataset1_feature_columns, dataset1_name, dataset1_frequency_hz, same_hr=True)
        self.lstm_loader_maker2 = PceLstmLoadersMaker(dfs_ds2, dataset2_feature_columns, dataset2_name, dataset2_frequency_hz, same_hr=True)

        self.discriminator_loaders_maker1 = PceDiscriminatorTripletLossLoadersMaker(dfs_ds1, dataset1_feature_columns, dataset1_name, dataset1_frequency_hz, same_hr=True)
        self.discriminator_loaders_maker2 = PceDiscriminatorTripletLossLoadersMaker(dfs_ds2, dataset2_feature_columns, dataset2_name, dataset2_frequency_hz, same_hr=True)

        self.lstm_name1 = dataset1_name + "PceLstm"
        self.lstm_name2 = dataset2_name + "PceLstm"
        self.discriminator_name1 = dataset1_name + "TripletDiscriminator"
        self.discriminator_name2 = dataset2_name + "TripletDiscriminator"


        self.lstm_computer_maker1 = PceLstmBatchComputerMaker(lstm_builder_cls, device, input_features_parameter_name, len(dataset1_feature_columns), additional_net_args1, name = self.lstm_name1)
        self.lstm_computer_maker2 = PceLstmBatchComputerMaker(lstm_builder_cls, device, input_features_parameter_name, len(dataset2_feature_columns), additional_net_args2, name = self.lstm_name2)

        self.discriminator_computer_maker1 = TripletDiscriminatorBatchComputerMaker(device, self.discriminator_name1)
        self.discriminator_computer_maker2 = TripletDiscriminatorBatchComputerMaker(device, self.discriminator_name2)

        self.device = device
        self.nepoch = nepoch

        self.frequency_hz1 = dataset1_frequency_hz
        self.frequency_hz2 = dataset2_frequency_hz

        self.additional_net_args1 = additional_net_args1
        self.additional_net_args2 = additional_net_args2

        self.label_ztransformer = self.discriminator_loaders_maker1.ztransformer

        self.ts_sub1 = ts_sub1
        self.ts_sub2 = ts_sub2
        self.val_sub1 = val_sub1
        self.val_sub2 = val_sub2

        self.main_index = main_index

    
    def train(
            self,
            lr,
            weight_decay,
            batch_size,
            ts_per_sample,
            ts_per_is,
            step_s,
            period_s,
            alpha,
            margin,
            **net_args
        ):
        args = locals()
        args.pop("self")
        args["ts_sub1"] = self.ts_sub1
        args["ts_sub2"] = self.ts_sub2
        args["val_sub1"] = self.val_sub1
        args["val_sub2"] = self.val_sub2
        args["main_index"] = self.main_index

        lstm_batch_computer1 = self.lstm_computer_maker1.make(period_s*self.frequency_hz1, ts_per_is, **net_args)
        lstm_batch_computer2 = self.lstm_computer_maker2.make(period_s*self.frequency_hz2, ts_per_is, **net_args)
        
        discriminator_batch_computer1 = self.discriminator_computer_maker1.make( lstm_batch_computer1.model, margin)
        discriminator_batch_computer2 = self.discriminator_computer_maker2.make( lstm_batch_computer2.model, margin)

        loaders_lstm1 = self.lstm_loader_maker1.make(batch_size, self.ts_sub1, ts_per_sample, ts_per_is, step_s, period_s)
        loaders_discriminator1 = self.discriminator_loaders_maker1.make(batch_size, self.ts_sub1, self.val_sub1, ts_per_sample, ts_per_is, step_s, period_s)

        loaders_lstm2 = self.lstm_loader_maker2.make(batch_size, self.ts_sub2, ts_per_sample, ts_per_is, step_s, period_s)
        loaders_discriminator2 = self.discriminator_loaders_maker2.make(batch_size, self.ts_sub2, self.val_sub2, ts_per_sample, ts_per_is, step_s, period_s)


        optimizer = torch.optim.Adam(
            [{'params': lstm_batch_computer1.model.parameters()},
             {'params': lstm_batch_computer2.model.parameters()}],
             lr=lr, weight_decay=weight_decay)
        
        metrics_computer_lstm = RegressionHR.TrainerJoint.MetricsComputerIS(self.label_ztransformer)
        metrics_computer_discriminator = DisplayCriterion.CosineSimilarityTripletLoss(margin=margin)

        model_output_mae = lambda o: metrics_computer_lstm.mae(o.label, o.prediction)        
        
        loaders_tr, loaders_ts, loaders_val = zip(loaders_lstm1, loaders_lstm2, loaders_discriminator1, loaders_discriminator2)
        batch_computers = [lstm_batch_computer1, lstm_batch_computer2, discriminator_batch_computer1, discriminator_batch_computer2]
        weights = [alpha, alpha, 1-alpha, 1-alpha]
        weights = [w/np.sum(weights) for w in weights]
        display_metrics = [model_output_mae, model_output_mae, metrics_computer_discriminator, metrics_computer_discriminator]


        batch_trainer = SequentialTrainer(batch_computers, optimizer, weights) # ([batch_computer_lstm, ], optimizer)
        epoch_trainer = ToolBox.MultiModelEpochTrainer(batch_trainer)

        train_helper = ToolBox.MultiModelTrainHelper(
            epoch_trainer, loaders_tr, loaders_val, loaders_ts,
            display_metrics, self.main_index
        )
            
        outputs = train_helper.train(self.nepoch)
        for train_name in [self.lstm_name1, self.lstm_name2]:
            outputs[train_name]["labels"] = metrics_computer_lstm.inverse_transform_label(outputs[train_name]['labels'])
            outputs[train_name]["predictions"] = metrics_computer_lstm.inverse_transform_label(outputs[train_name]['predictions'])
        return {
            **{
            "args": args,
            "run_class": self.__class__.__name__
            }, **outputs
        }




class PceLstmTripletDiscriminatorPamap2DaliaJointTraining(DoubleDatasetPceLstmTripletDiscriminatorFullTrainer):
    def __init__(
        self,
        dfs_ds1,
        dfs_ds2,
        device,
        nepoch,
        ts_sub1,
        ts_sub2,
        val_sub1,
        val_sub2,
        main_index,
        dataset1_name = "pamap2",
        dataset1_feature_columns = [
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
            'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
            'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
            'a_ymag', 'a_zmag'
        ],
        dataset2_name = "dalia",
        dataset2_feature_columns = [
            'heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
            'chest-ACC-0','chest-ACC-1', 'chest-ACC-2'
        ],
        dataset1_frequency_hz = 100,
        dataset2_frequency_hz = 32,
        input_features_parameter_name = "nattrs",
        lstm_builder_cls = RegressionHR.PceLstmModel.make_par_enc_pce_lstm,
        additional_net_args1 = dict(),
        additional_net_args2 = dict(),
        ):

        super().__init__(
            dfs_ds1 = dfs_ds1,
            dfs_ds2 = dfs_ds2,
            dataset1_name = dataset1_name,
            dataset1_feature_columns = dataset1_feature_columns,
            dataset2_name = dataset2_name,
            dataset2_feature_columns = dataset2_feature_columns,
            dataset1_frequency_hz = dataset1_frequency_hz,
            dataset2_frequency_hz = dataset2_frequency_hz,
            device = device,
            nepoch = nepoch,
            input_features_parameter_name = input_features_parameter_name,
            ts_sub1 = ts_sub1,
            ts_sub2 = ts_sub2,
            val_sub1 = val_sub1,
            val_sub2 = val_sub2,
            main_index = main_index,
            lstm_builder_cls = lstm_builder_cls,
            additional_net_args1 = additional_net_args1,
            additional_net_args2 = additional_net_args2,
        )





class PceLstmPamap2TripletDiscriminator(PceLstmTripletDiscriminator):
    def __init__(self, dfs, device, nepoch, additional_args = dict(), additional_net_args = dict()):

        super().__init__(
            dfs = dfs,
            device = device,
            nepoch = nepoch,
            lstm_builder_cls = RegressionHR.PceLstmModel.make_par_enc_pce_lstm,
            feature_columns = [
                'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
                'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
                'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
                'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
                'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
                'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
                'a_ymag', 'a_zmag'
            ],
            dataset_name = "pamap2",
            frequency_hz = 100,
            input_features_parameter_name = "nattrs",
            additional_args = additional_args,
            additional_net_args = additional_net_args,
            )


class PceLstmDaliaTripletDiscriminator(PceLstmTripletDiscriminator):
    def __init__(self, dfs, device, nepoch,
                 feature_columns = [
                    'heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                    'chest-ACC-0','chest-ACC-1', 'chest-ACC-2'
                 ], additional_args = dict(), additional_net_args = dict()):

        super().__init__(
            dfs = dfs,
            device = device,
            nepoch = nepoch,
            lstm_builder_cls = RegressionHR.PceLstmModel.make_par_enc_pce_lstm,
            feature_columns = feature_columns,
            dataset_name = "dalia",
            frequency_hz = 32,
            input_features_parameter_name = "nattrs",
            additional_args = additional_args,
            additional_net_args = additional_net_args,
            )


class PceLstmPamap2FullTrainerJointValidation(SingleNetFullTrainerJointValidationIS):
    def __init__(self, dfs, device, nepoch, feature_columns=[
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
            'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
            'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
            'a_ymag', 'a_zmag'
        ], additional_args = dict(), additional_net_args = dict(), model_name = None):
        super().__init__(
            dfs, device, nepoch, RegressionHR.PceLstmModel.make_par_enc_pce_lstm , 
            RegressionHR.Preprocessing.PceLstmTransformerGetter, "pamap2", feature_columns,
             100, "nattrs", additional_args, additional_net_args, model_name)


class NoPceLstmPamap2FullTrainerJointValidation(SingleNetFullTrainerJointValidationIS):
    def __init__(self, dfs, device, nepoch, feature_columns=[
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
            'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
            'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
            'a_ymag', 'a_zmag'
        ], additional_args = dict(), additional_net_args = dict(), model_name = None):
        super(NoPceLstmPamap2FullTrainerJointValidation, self).__init__(
            dfs, device, nepoch, RegressionHR.PceLstmModel.make_par_enc_no_pce_lstm , 
            RegressionHR.Preprocessing.PceLstmTransformerGetter, "pamap2", feature_columns,
             100, "nattrs", additional_args, additional_net_args, model_name)