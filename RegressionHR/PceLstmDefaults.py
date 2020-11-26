import numpy as np
from preprocessing_utils import (
    HZMeanSubstitute, ZTransformer2, FFT, FFT_KEEP,
    TimeSnippetAggregator, FeatureLabelSplit,
    TransformerPipeline, SampleMaker, NoDiffInitialStatePredictionSplit,
    RecursiveHrMasker, Downsampler, IdentityTransformer, LocalMeanReplacer, LinearImputation,
    ApplyTransformer, PceDecoderLoaderTransformer, SlidingWindow)


class PamapPreprocessingTransformerGetter():
    def __init__(self):
        self.ztransformer = ZTransformer2([
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr',
            'h_zgyr', 'h_xmag', 'h_ymag', 'h_zmag', 'c_temperature',
            'c_xacc16', 'c_yacc16', 'c_zacc16', 'c_xacc6', 'c_yacc6',
            'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16',
            'a_zacc16', 'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr',
            'a_ygyr', 'a_zgyr', 'a_xmag', 'a_ymag', 'a_zmag'
        ], dataset="pamap2")

    def __call__(self, ts_per_sample=30, ts_per_is=2, frequency_hz=100, period_s=4, step_s=2, sample_step_ratio = 0.5):

        feature_columns = [
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
            'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
            'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
            'a_ymag', 'a_zmag'
        ]

        self.hr_lin_imputation = LinearImputation("heart_rate")

        self.local_mean_imputer = LocalMeanReplacer() 

        meansub = HZMeanSubstitute()

        feature_label_splitter = FeatureLabelSplit(
            label_column = "heart_rate",
            feature_columns = feature_columns
        )

        recursive_hr_masker = RecursiveHrMasker(0)

        sample_maker = SampleMaker(ts_per_sample, int(ts_per_sample*sample_step_ratio))

        is_pred_split = NoDiffInitialStatePredictionSplit(ts_per_sample, ts_per_is)

        ts_aggregator = TimeSnippetAggregator(size=int(frequency_hz*period_s),
                                              step=int(frequency_hz*step_s))


        return TransformerPipeline(
            self.ztransformer,
            self.hr_lin_imputation,
            self.local_mean_imputer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            sample_maker,
            is_pred_split,
            recursive_hr_masker
        )



class PamapPceDecoderPreprocessingTransformerGetter():
    def __init__(self):
        self.ztransformer = ZTransformer2([
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr',
            'h_zgyr', 'h_xmag', 'h_ymag', 'h_zmag', 'c_temperature',
            'c_xacc16', 'c_yacc16', 'c_zacc16', 'c_xacc6', 'c_yacc6',
            'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16',
            'a_zacc16', 'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr',
            'a_ygyr', 'a_zgyr', 'a_xmag', 'a_ymag', 'a_zmag'
        ], dataset="pamap2")

    def __call__(self, ts_per_is=2, frequency_hz=100, period_s=4, step_s=2, sample_step_ratio = 1):

        feature_columns = [
            'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
            'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
            'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
            'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
            'a_ymag', 'a_zmag'
        ]

        self.hr_lin_imputation = LinearImputation("heart_rate")

        self.local_mean_imputer = LocalMeanReplacer(mean_width=frequency_hz) 

        meansub = HZMeanSubstitute()

        feature_label_splitter = FeatureLabelSplit(
            label_column = "heart_rate",
            feature_columns = feature_columns
        )

        sample_maker = SlidingWindow(ts_per_is, int(ts_per_is*sample_step_ratio))

        ts_aggregator = TimeSnippetAggregator(size=int(frequency_hz*period_s),
                                              step=int(frequency_hz*step_s))
        reshape = ApplyTransformer(lambda v: (np.swapaxes(v[0].squeeze(2), 2,3), v[1])) 
      
        tpipe  = TransformerPipeline(
            self.ztransformer,
            self.hr_lin_imputation,
            self.local_mean_imputer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            sample_maker,
            reshape
        )

        return PceDecoderLoaderTransformer(tpipe)
