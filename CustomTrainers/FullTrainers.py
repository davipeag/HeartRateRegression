import torch
import numpy as np
import math
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

from Constants import DatasetMapping

import Trainer
from Trainer import BatchTrainers
from Trainer import BatchComputers
from Trainer import ToolBox
from Trainer import DisplayCriterion

from Trainer.BatchComputers import BatchComputerPceDiscriminator, BatchComputerXY, BatchComputerIS, BatchComputerTripletLoss
from Trainer.BatchTrainers import SequentialTrainer

import Models
from Models import BaseModels

import CustomLoss

import sklearn.metrics

# from torch.nn.functional import sigmoid
from torch import sigmoid
from PreprocessingHelpers.TransformerGetters import (
  PceDiscriminatorTransformerGetter,
  PceLstmTransformerGetter
)

from Constants import DatasetMapping


# class PceLstmDeepDiscriminatorFullTrainer():
#     def __init__(
#         self,
#         dfs,
#         device,
#         nepoch,
        
#         dataset_name,
#         feature_columns,
#         frequency_hz,
#         input_features_parameter_name,

#         lstm_transformer_getter_cls,
#         discriminator_transformer_getter_cls,

#         additional_args = dict(),
#         args_to_net_args_mapping = dict(),
#         args_function_mapping = dict(),
#         lstm_model_name = None,
#         discriminator_model_name = None
#         ):
#         self.dfs = dfs
#         self.device = device
#         self.nepoch = nepoch

#         self.transformers_lstm = lstm_transformer_getter_cls(feature_columns, dataset_name)
#         self.transformers_discriminator = discriminator_transformer_getter_cls(feature_columns, dataset_name)
        
#         self.nets_builder_cls = RegressionHR.PceLstmModel.make_pce_lstm_and_discriminator
        
#         self.feature_columns = feature_columns
#         self.frequency_hz = frequency_hz
#         self.input_features_parameter_name = input_features_parameter_name
#         self.additional_args = additional_args
#         self.add_to_net_args_mapping = args_to_net_args_mapping
#         self.args_function_mapping = args_function_mapping
#         self.lstm_model_name = lstm_model_name
#         self.discriminator_model_name = discriminator_model_name
#         self.frequency_hz_in = DatasetMapping.FrequencyMapping[dataset_name]
    
#     def train(
#         self,
#         lr,
#         weight_decay,
#         batch_size,
#         ts_sub,
#         ts_per_window,
#         ts_per_is,
#         period_s,
#         step_s,
#         **net_args
#         ):
#         frequency_hz = self.frequency_hz
#         args = locals()
#         for k,func in self.args_function_mapping.items():
#             args[k] = func(args)
#         net_args[self.input_features_parameter_name] = len(self.feature_columns)
#         for k,v in self.add_to_net_args_mapping.items():
#             net_args[k] = args[v]
        
#         args = {**args, **self.additional_args, **net_args}
#         args.pop("self")

#         disciminator_args = {

#         }
        
        
#         ldf = len(self.dfs[ts_sub])
#         ts_per_window_ts = int(ldf/(self.frequency_hz_in*period_s))-3
#         transformers_ts = self.transformers(
#             period_s = period_s, frequency_hz = frequency_hz,
#             ts_per_window=ts_per_window_ts, ts_per_is=ts_per_is,
#             step_s = step_s, sample_step_ratio = 1)

#         # print(net_args)
#         net = self.net_builder_cls(**net_args).to(self.device)
#         PPG.Models.initialize_weights(net)

#         criterion = torch.nn.L1Loss().to(self.device)
        
#         optimizer = torch.optim.Adam(net.parameters(), lr=lr,
#                                      weight_decay=weight_decay)

#         batch_computer = BatchComputerIS(net, criterion, self.device, self.model_name)
#         batch_trainer = SequentialTrainer([batch_computer], optimizer)

        
        
#         ztransformer = self.transformers.ztransformer
#         metrics_computer = RegressionHR.TrainerJoint.MetricsComputerIS(ztransformer)
        
#         transformers_tr = self.transformers(period_s=period_s, ts_per_window = ts_per_window, frequency_hz=frequency_hz,
#                                             ts_per_is=ts_per_is, sample_step_ratio=1, step_s = step_s)

    
#         loader_tr, loader_val, loader_ts = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
#             transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
#             dataset_cls=PPG.UtilitiesDataXY.ISDataset
#         ).make_loaders(ts_sub, 0.8)

#         epoch_trainer = ToolBox.MultiModelEpochTrainer(batch_trainer)
        
#         train_helper = ToolBox.MultiModelTrainHelper(
#             epoch_trainer, [loader_tr], [loader_val], [loader_ts],
#             [lambda o: metrics_computer.mae(o.label, o.prediction)]
#         )
            
#         outputs = train_helper.train(self.nepoch)
#         outputs = list(outputs.values())[0]

#         outputs["labels"] = metrics_computer.inverse_transform_label(outputs['labels'])
#         outputs["predictions"] = metrics_computer.inverse_transform_label(outputs['predictions'])
#         return {
#             **{
#             "args": args,
#             "run_class": self.__class__.__name__
#             }, **outputs
#         }



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
        input_features_parameter_name,
        frequency_hz = None,
        additional_args = dict(),
        args_to_net_args_mapping = dict(),
        args_function_mapping = dict(),
        model_name = None

        ):
        self.dfs = dfs
        self.device = device
        self.transformers = transformer_getter_cls(feature_columns, dataset_name)
        self.nepoch = nepoch
        self.net_builder_cls = net_builder_cls
        self.feature_columns = feature_columns
        if frequency_hz is None:
            frequency_hz = DatasetMapping.FrequencyMapping[dataset_name]
        self.frequency_hz = frequency_hz
        self.input_features_parameter_name = input_features_parameter_name
        self.additional_args = additional_args
        self.add_to_net_args_mapping = args_to_net_args_mapping
        self.args_function_mapping = args_function_mapping
        self.model_name = None
        self.frequency_hz_in = DatasetMapping.FrequencyMapping[dataset_name]
    
    def train(
        self,
        lr,
        weight_decay,
        batch_size,
        ts_sub,
        ts_per_window,
        ts_per_is,
        period_s,
        step_s = None,
        **net_args
        ):
        if step_s is None:
            step_s = period_s
        frequency_hz = self.frequency_hz
        args = locals()
        for k,func in self.args_function_mapping.items():
            args[k] = func(args)
        net_args[self.input_features_parameter_name] = len(self.feature_columns)
        for k,v in self.add_to_net_args_mapping.items():
            net_args[k] = args[v]
        
        args = {**args, **self.additional_args, **net_args}
        args.pop("self")
        
        
        ldf = len(self.dfs[ts_sub])
        ts_per_window_ts = int(ldf/(self.frequency_hz_in*period_s))-3
        transformers_ts = self.transformers(
            period_s = period_s, frequency_hz = frequency_hz,
            ts_per_window=ts_per_window_ts, ts_per_is=ts_per_is,
            step_s = step_s, window_step_ratio = 1)

        print(net_args)
        net = self.net_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(net)

        criterion = torch.nn.L1Loss().to(self.device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        batch_computer = BatchComputerIS(net, criterion, self.device, self.model_name)
        batch_trainer = SequentialTrainer([batch_computer], optimizer)

        
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = RegressionHR.TrainerJoint.MetricsComputerIS(ztransformer)
        
        transformers_tr = self.transformers(period_s=period_s, ts_per_window = ts_per_window, frequency_hz=frequency_hz,
                                            ts_per_is=ts_per_is, window_step_ratio=1, step_s = step_s)

    
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

        outputs["labels"] = metrics_computer.inverse_transform_label(outputs['labels'])
        outputs["predictions"] = metrics_computer.inverse_transform_label(outputs['predictions'])
        return {
            **{
            "args": args,
            "run_class": self.__class__.__name__
            }, **outputs
        }


class PceDeepDiscriminatorAndLstmFullTrainerJointValidationIS():
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        nets_builder_cls,
        lstm_transformer_getter_cls,
        discriminator_transformer_getter_cls,
        dataset_name,
        feature_columns,
        input_features_parameter_name,
        frequency_hz = None,
        additional_args = dict(),
        args_to_net_args_mapping = dict(),
        args_function_mapping = dict(),
        model_name = "",

        ):
        self.dfs = dfs
        self.device = device
        self.transformers_lstm = lstm_transformer_getter_cls(feature_columns, dataset_name)
        self.transformers_discriminator = discriminator_transformer_getter_cls(feature_columns, dataset_name)
        self.nepoch = nepoch
        self.nets_builder_cls = nets_builder_cls
        self.feature_columns = feature_columns
        if frequency_hz is None:
            frequency_hz = DatasetMapping.FrequencyMapping[dataset_name]
        self.frequency_hz = frequency_hz
        self.input_features_parameter_name = input_features_parameter_name
        self.additional_args = additional_args
        self.add_to_net_args_mapping = args_to_net_args_mapping
        self.args_function_mapping = args_function_mapping
        self.model_name = model_name
        self.frequency_hz_in = DatasetMapping.FrequencyMapping[dataset_name]
    
    def train(
        self,
        lr,
        weight_decay,
        batch_size,
        ts_sub,
        val_sub,
        ts_per_window,
        ts_per_is,
        period_s,
        step_s,
        alpha,
        **net_args
        ):
        if step_s is None:
            step_s = period_s
        frequency_hz = self.frequency_hz
        args = locals()
        for k,func in self.args_function_mapping.items():
            args[k] = func(args)
        net_args[self.input_features_parameter_name] = len(self.feature_columns)
        for k,v in self.add_to_net_args_mapping.items():
            net_args[k] = args[v]
        
        args = {**args, **self.additional_args, **net_args}
        args.pop("self")
        
        
        ldf = len(self.dfs[ts_sub])
        ts_per_window_ts = int(ldf/(self.frequency_hz_in*period_s))-3
        transformers_ts_lstm = self.transformers_lstm(
            period_s = period_s, frequency_hz = frequency_hz,
            ts_per_window=ts_per_window_ts, ts_per_is=ts_per_is,
            step_s = step_s, window_step_ratio = 1)

        print(net_args)
        nets = self.nets_builder_cls(**net_args)
        [n.to(self.device) for n in nets]
        lstm, discriminator = nets
        PPG.Models.initialize_weights(lstm)
        PPG.Models.initialize_weights(discriminator)

        criterion1 = torch.nn.L1Loss().to(self.device)
        criterion2 = torch.nn.BCEWithLogitsLoss().to(self.device)
        
        optimizer = torch.optim.Adam([{"params": lstm.parameters()},
                                      {"params": discriminator.discriminator.parameters()}], lr=lr,
                                     weight_decay=weight_decay)
        
        batch_computer1 = BatchComputerIS(lstm, criterion1, self.device, self.model_name + "_LSTM")
        batch_computer2 = BatchComputerPceDiscriminator(discriminator, criterion2, self.device, self.model_name + "_Discriminator")
        batch_trainer = SequentialTrainer([batch_computer1, batch_computer2], optimizer, weights=[alpha, 1-alpha])

        
        
        ztransformer = self.transformers_lstm.ztransformer
        metrics_computer = RegressionHR.TrainerJoint.MetricsComputerIS(ztransformer)
        
        transformers_tr_lstm = self.transformers_lstm(period_s=period_s, ts_per_window = ts_per_window, frequency_hz=frequency_hz,
                                            ts_per_is=ts_per_is, window_step_ratio=1, step_s = step_s)

        

        loader_tr1, loader_val1, loader_ts1 = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr_lstm, transformers_ts=transformers_ts_lstm, dfs = self.dfs, batch_size_tr=batch_size,
            dataset_cls=PPG.UtilitiesDataXY.ISDataset
        ).make_loaders(ts_sub, 0.8)

        sample_step_ratio = ts_per_window/ts_per_is

        transformers2 = self.transformers_discriminator(
            period_s=period_s, step_s=step_s, frequency_hz = frequency_hz,
            sample_step_ratio=sample_step_ratio, ts_per_is=ts_per_is)
        
        

        loader_tr2, loader_val2, loader_ts2 = RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers2, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)


        epoch_trainer = ToolBox.MultiModelEpochTrainer(batch_trainer)
        def sigmoid(x): return 1 / (1 + math.exp(-x))
        
        def accuracy(o): (np.sum((sigmoid(o.prediction) > 0.5)== o.label)/len(o.prediction)) 

        train_helper = ToolBox.MultiModelTrainHelper(
            epoch_trainer, [loader_tr1, loader_tr2], [loader_val1, loader_val2], [loader_ts1, loader_ts2],
            [lambda o: metrics_computer.mae(o.label, o.prediction), accuracy]
        )
            
        outputs = train_helper.train(self.nepoch)
        # outputs = list(outputs.values())[0]

        # outputs["labels"] = metrics_computer.inverse_transform_label(outputs['labels'])
        # outputs["predictions"] = metrics_computer.inverse_transform_label(outputs['predictions'])
        return {
            **{
            "args": args,
            "run_class": self.__class__.__name__
            }, **outputs
        }




class SingleNetFullTrainerJointValidationXY():
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
        args_to_net_args_mapping = dict(),
        args_function_mapping = dict(),
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
        self.add_to_net_args_mapping = args_to_net_args_mapping
        self.args_function_mapping = args_function_mapping
        self.model_name = None
        self.frequency_hz_in = DatasetMapping.FrequencyMapping[dataset_name]
    
    def train(
        self,
        lr,
        weight_decay,
        batch_size,
        ts_sub,
        ts_per_window,
        ts_per_is,
        period_s,
        step_s = None,
        **net_args
        ):
        if step_s is None:
            step_s = period_s
        frequency_hz = self.frequency_hz
        args = locals()
        for k,func in self.args_function_mapping.items():
            args[k] = func(args)
        net_args[self.input_features_parameter_name] = len(self.feature_columns)
        for k,v in self.add_to_net_args_mapping.items():
            net_args[k] = args[v]
        
        args = {**args, **self.additional_args, **net_args}
        args.pop("self")
        
        
        ldf = len(self.dfs[ts_sub])
        ts_per_window_ts = int(ldf/(self.frequency_hz_in*period_s))-3
        transformers_ts = self.transformers(
            period_s = period_s, frequency_hz = frequency_hz,
            ts_per_window=ts_per_window_ts, ts_per_is=ts_per_is,
            step_s = step_s, sample_step_ratio = 1)

        # print(net_args)
        net = self.net_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(net)

        criterion = torch.nn.L1Loss().to(self.device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        batch_computer = BatchComputerXY(net, criterion, self.device, self.model_name)
        batch_trainer = SequentialTrainer([batch_computer], optimizer)

        
        
        ztransformer = self.transformers.ztransformer
        metrics_computer = RegressionHR.TrainerJoint.MetricsComputerIS(ztransformer)
        
        transformers_tr = self.transformers(period_s=period_s, ts_per_window = ts_per_window, frequency_hz=frequency_hz,
                                            ts_per_is=ts_per_is, sample_step_ratio=1, step_s = step_s)

    
        loader_tr, loader_val, loader_ts = PPG.UtilitiesDataXY.JointTrValDataLoaderFactory(
            transformers_tr, transformers_ts=transformers_ts, dfs = self.dfs, batch_size_tr=batch_size,
            dataset_cls=PPG.UtilitiesDataXY.XYDataset2
        ).make_loaders(ts_sub, 0.8)

        epoch_trainer = ToolBox.MultiModelEpochTrainer(batch_trainer)
        
        train_helper = ToolBox.MultiModelTrainHelper(
            epoch_trainer, [loader_tr], [loader_val], [loader_ts],
            [lambda o: metrics_computer.mae(o.label, o.prediction)]
        )
            
        outputs = train_helper.train(self.nepoch)
        outputs = list(outputs.values())[0]

        outputs["labels"] = metrics_computer.inverse_transform_label(outputs['labels'])
        outputs["predictions"] = metrics_computer.inverse_transform_label(outputs['predictions'])
        return {
            **{
            "args": args,
            "run_class": self.__class__.__name__
            }, **outputs
        }

