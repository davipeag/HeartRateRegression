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

from Trainer.BatchComputers import BatchComputerIS
from Trainer.BatchTrainers import SequentialTrainer


import Models
from Models import BaseModels

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

        return {
            **{
            "args": args,
            "run_class": self.__class__.__name__
            }, **outputs
        }


class NoPceLstmPamap2FullTrainerJointValidationImuOnly(SingleNetFullTrainerJointValidationIS):
    def __init__(self, dfs, device, nepoch, feature_columns=[
            'heart_rate', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 
            'c_xacc16', 'c_yacc16', 'c_zacc16','c_xacc6', 'c_yacc6', 'c_zacc6',
            'c_xgyr', 'c_ygyr', 'c_zgyr', 'a_xacc16', 'a_yacc16', 'a_zacc16',
            'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr'
        ]):
        super(NoPceLstmPamap2FullTrainerJointValidationImuOnly, self).__init__(
            dfs, device, nepoch, RegressionHR.PceLstmModel.make_par_enc_pce_lstm , 
            RegressionHR.Preprocessing.PceLstmTransformerGetter, "pamap2", feature_columns,
             100, "nattrs")