import numpy as np
from preprocessing_utils import (
    HZMeanSubstitute, ZTransformer2, FFT, FFT_KEEP,
    TimeSnippetAggregator, FeatureLabelSplit,
    TransformerPipeline, SampleMaker, NoDiffInitialStatePredictionSplit,
    RecursiveHrMasker, Downsampler, IdentityTransformer, LocalMeanReplacer, LinearImputation,
    ApplyTransformer, PceDecoderLoaderTransformer, SlidingWindow, TripletPCELoaderTransformer, XYMasker)

from Constants import DatasetMapping



class DeepConvLstmTransformerGetter():

    def __init__(self, feature_columns, dataset_name, same_hr=False, label_column = "heart_rate"):
        self.feature_columns = feature_columns
        self.ztransformer = ZTransformer2(
            self.feature_columns, dataset=dataset_name, same_hr=same_hr)
        self.label_column = "heart_rate"
        self.frequency_hz_in = DatasetMapping.FrequencyMapping[dataset_name]

    def __call__(self, ts_per_window, ts_per_is, period_s, frequency_hz, sample_step_ratio = 1):

        if self.frequency_hz_in < 2*frequency_hz:
            downsampler = IdentityTransformer()
        else:
            downsampler = Downsampler(frequency_hz/self.frequency_hz_in)
        
        feature_columns = self.feature_columns 
        feature_count = len(self.feature_columns)
        meansub = HZMeanSubstitute()

        sample_per_ts = int(frequency_hz*period_s)

        feature_label_splitter = FeatureLabelSplit(
            label_column = self.label_column,
            feature_columns = feature_columns
        )


        recursive_hr_masker = XYMasker(0, ts_per_is)

        sample_maker = SampleMaker(ts_per_window, int(ts_per_window*sample_step_ratio))

        ts_aggregator = TimeSnippetAggregator(size=sample_per_ts, step=sample_per_ts)

        reshape = ApplyTransformer(lambda v:  (v[0].reshape([-1, 1, ts_per_window*sample_per_ts, feature_count]), v1[1])) 

        return TransformerPipeline(
            downsampler,
            self.ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            sample_maker,
            recursive_hr_masker,
            reshape
        )

        # return TransformerPipeline(
        #     self.ztransformer,
        #     feature_label_splitter,
        #     ts_aggregator,
        #     meansub,
        #     sample_maker,
        #     is_pred_split,
        #     recursive_hr_masker
        # )




class PceLstmTransformerGetter():
    def __init__(self, feature_columns, dataset_name, same_hr=False):
        self.feature_columns = feature_columns
        self.ztransformer = ZTransformer2(
            self.feature_columns, dataset=dataset_name, same_hr=same_hr)

    def __call__(self, ts_per_sample, ts_per_is, frequency_hz, period_s, step_s, sample_step_ratio):

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

