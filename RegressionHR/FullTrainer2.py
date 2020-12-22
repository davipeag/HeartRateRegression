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