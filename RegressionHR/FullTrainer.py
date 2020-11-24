import torch
import copy

import PPG
import RegressionHR
from PPG import UtilitiesDataXY

from PPG.Models import (SnippetConvolutionalTransformer, initialize_weights)
from PPG.TrainerXY import (EpochTrainerXY, MetricsComputerXY, TrainHelperXY)
from PPG.TrainerIS import (EpochTrainerIS, MetricsComputerIS, TrainHelperIS)
from preprocessing_utils import ZTransformer2


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
        ts_per_samples = [30, 60]
    ):
        args = locals()
        args.pop("self")
        net_args = copy.deepcopy(args)
        [net_args.pop(v) for v in ("ts_sub", "val_sub", "lr", "weight_decay", "batch_size", "ts_per_samples")]
        frequency_hz = 100
        period_s = 4
        step_s = 2
        
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