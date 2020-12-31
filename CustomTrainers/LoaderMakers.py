import PreprocessingHelpers
from PreprocessingHelpers import TransformerGetters
import PPG
import RegressionHR

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
        self.transformers_lstm = TransformerGetters.PceLstmTransformerGetter(feature_columns, dataset_name, same_hr=same_hr)
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


class PceDiscriminatorLoadersMaker():
    def __init__(
            self,
            dfs,
            feature_columns,
            dataset_name,
            frequency_hz,
            same_hr = False
        ):
        self.dfs = dfs
        self.transformer_getter = TransformerGetters.PceDiscriminatorTransformerGetter(feature_columns, dataset_name, same_hr=same_hr, false_label=0)
        self.ztransformer = self.transformer_getter.ztransformer
        self.frequency_hz = frequency_hz
    
    def make(self, batch_size, ts_sub, val_sub, ts_per_is, step_s, period_s, sample_step_ratio):
            
        transformers = self.transformer_getter(
            period_s=period_s, step_s=step_s, frequency_hz = self.frequency_hz, sample_step_ratio=sample_step_ratio)
        
        return RegressionHR.UtilitiesData.PceDiscriminatorDataLoaderFactory(
            transformers, self.dfs, batch_size_tr=batch_size).make_loaders(ts_sub, val_sub)
