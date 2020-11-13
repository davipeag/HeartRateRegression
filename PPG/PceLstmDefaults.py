
from preprocessing_utils import (
    HZMeanSubstitute, ZTransformer2, FFT, FFT_KEEP,
    TimeSnippetAggregator, FeatureLabelSplit,
    TransformerPipeline, SampleMaker, NoDiffInitialStatePredictionSplit,
    RecursiveHrMasker, Downsampler)


class PreprocessingTransformerGetter():
    def __init__(self):
        self.ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])

    def __call__(self, ts_per_sample=30, ts_per_is=2, frequency_hz=32, period_s=8, step_s=2):

        feature_columns = [
                        'heart_rate',
                        'wrist-ACC-0',
                        'wrist-ACC-1',
                        'wrist-ACC-2',
                        'wrist-BVP-0',
                        ]

        BVP = 'wrist-BVP-0'
        BVP_IDX = feature_columns.index(BVP)
    
        meansub = HZMeanSubstitute()


        fft = FFT(BVP_IDX)

        feature_label_splitter = FeatureLabelSplit(
            label_column = "heart_rate",
            feature_columns = feature_columns
        )

        recursive_hr_masker = RecursiveHrMasker(0)

        sample_maker = SampleMaker(ts_per_sample, ts_per_sample)

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
            recursive_hr_masker,
            fft
        )


class IeeePreprocessingTransformerGetter():
    def __init__(self, donwsampling_ratio=32/125):
        self.ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-BVP-1'], dataset= "ieee_train")
        
        
        self.downsampler = Downsampler(donwsampling_ratio)
        self.downsampling_ratio = donwsampling_ratio

    def __call__(self, ts_per_sample=30, ts_per_is=2, frequency_hz=32,
                                        period_s=8, step_s=2):
        frequency_hz = self.downsampling_ratio * 125
        feature_columns = [
                        'heart_rate',
                        'wrist-ACC-0',
                        'wrist-ACC-1',
                        'wrist-ACC-2',
                        'wrist-BVP-0',
                        'wrist-BVP-1',
                        ]
        BVP_IDX = [feature_columns.index('wrist-BVP-0'),feature_columns.index('wrist-BVP-1')]
    
        meansub = HZMeanSubstitute()

        
        fft = FFT(BVP_IDX)

        feature_label_splitter = FeatureLabelSplit(
            label_column = "heart_rate",
            feature_columns = feature_columns
        )

        recursive_hr_masker = RecursiveHrMasker(0)

        sample_maker = SampleMaker(ts_per_sample, ts_per_sample//2)

        is_pred_split = NoDiffInitialStatePredictionSplit(ts_per_sample, ts_per_is)

        ts_aggregator = TimeSnippetAggregator(size=int(frequency_hz*period_s), step=int(frequency_hz*step_s))


        return TransformerPipeline(
            self.downsampler,
            self.ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            sample_maker,
            is_pred_split,
            recursive_hr_masker,
            fft
        )
