import numpy as np
from preprocessing_utils import (
    HZMeanSubstitute, ZTransformer2, FFT, FFT_KEEP,
    TimeSnippetAggregator, FeatureLabelSplit,
    TransformerPipeline, SampleMaker, NoDiffInitialStatePredictionSplit,
    RecursiveHrMasker, Downsampler, IdentityTransformer, LocalMeanReplacer, LinearImputation,
    ApplyTransformer, PceDecoderLoaderTransformer, SlidingWindow, TripletPCELoaderTransformer)


class FFNNPreprocessingTransformerGetter():
    def __init__(self, feature_columns, dataset_name):
        self.feature_columns = feature_columns
        self.ztransformer = ZTransformer2(
            self.feature_columns, dataset=dataset_name)

    def __call__(self, ts_per_sample=30, frequency_hz=100, period_s=4, step_s=2, sample_step_ratio=1, ts_per_is = 1):

        if ts_per_is != 1:
            raise ValueError(f"Parameter ts per is needs to be 1, but {ts_per_is}")

        self.hr_lin_imputation = LinearImputation("heart_rate")
        self.local_mean_imputer = LocalMeanReplacer()

        meansub = HZMeanSubstitute()
        feature_label_splitter = FeatureLabelSplit(
            label_column="heart_rate",
            feature_columns=self.feature_columns
        )

        recursive_hr_masker = RecursiveHrMasker(0)

        sample_maker = SampleMaker(
            ts_per_sample, int(ts_per_sample*sample_step_ratio))

        is_pred_split = NoDiffInitialStatePredictionSplit(
            ts_per_sample, ts_per_is)

        ts_aggregator = TimeSnippetAggregator(size=int(frequency_hz*period_s),
                                              step=int(frequency_hz*step_s))

        compute_mean = ApplyTransformer(
            lambda vs: [np.mean(v, axis=-1) for v in vs])

        return TransformerPipeline(
            self.ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            sample_maker,
            is_pred_split,
            recursive_hr_masker,
            compute_mean
        )


class PceLstmTransformerGetter():
    def __init__(self, feature_columns, dataset_name, same_hr=False):
        self.feature_columns = feature_columns
        self.ztransformer = ZTransformer2(
            self.feature_columns, dataset=dataset_name, same_hr=same_hr)

    def __call__(self, ts_per_sample=30, ts_per_is=2, frequency_hz=100, period_s=4, step_s=2, sample_step_ratio = 1):

        feature_columns = self.feature_columns 

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
            feature_label_splitter,
            ts_aggregator,
            meansub,
            sample_maker,
            is_pred_split,
            recursive_hr_masker
        )


class PceLstmDecoderTripletLossTransformerGetter():
    def __init__(self, feature_columns, dataset, frequency_hz, label_column = "heart_rate", same_hr=False):
        self.ztransformer = ZTransformer2(feature_columns, dataset=dataset, same_hr=same_hr)
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.frequency_hz = frequency_hz

    def __call__(self, ts_per_is=2, period_s=4, step_s=2, sample_step_ratio = 1):

        feature_columns = self.feature_columns
        frequency_hz = self.frequency_hz

        meansub = HZMeanSubstitute()

        feature_label_splitter = FeatureLabelSplit(
            label_column = self.label_column,
            feature_columns = feature_columns
        )

        sample_maker = SlidingWindow(ts_per_is, int(ts_per_is*sample_step_ratio))

        ts_aggregator = TimeSnippetAggregator(size=int(frequency_hz*period_s),
                                              step=int(frequency_hz*step_s))
        reshape = ApplyTransformer(lambda v: (np.swapaxes(v[0].squeeze(2), 2,3), v[1])) 
      
        transformer  = TransformerPipeline(
            self.ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            sample_maker,
            reshape
        )

        return TripletPCELoaderTransformer(transformer)


