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

from Trainer.BatchComputers import BatchComputerXY, BatchComputerIS, BatchComputerTripletLoss
from Trainer.BatchTrainers import SequentialTrainer

import Models
from Models import BaseModels

import CustomLoss

import sklearn.metrics

# from torch.nn.functional import sigmoid
from torch import sigmoid


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
    
    def train(
        self,
        lr,
        weight_decay,
        batch_size,
        ts_sub,
        ts_per_window,
        ts_per_is,
        period_s,
        **net_args
    ):
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
        ts_per_window_ts = int(ldf/(frequency_hz))-3
        transformers_ts = self.transformers(
            period_s = period_s, frequency_hz = frequency_hz,
            ts_per_window=ts_per_window_ts, ts_per_is=ts_per_is)

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
                                            ts_per_is=ts_per_is)

    
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

