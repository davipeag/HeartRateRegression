
from preprocessing_utils import (
    HZMeanSubstitute, ZTransformer2, FFT,
    TimeSnippetAggregator, FeatureLabelSplit,
    TransformerPipeline, SampleMaker, NoDiffInitialStatePredictionSplit,
    RecursiveHrMasker, Downsampler)

def get_preprocessing_transformer(ts_per_sample=30, ts_per_is=2, frequency_hz=32, period_s=8):

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

    ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])

    fft = FFT(BVP_IDX)

    feature_label_splitter = FeatureLabelSplit(
        label_column = "heart_rate",
        feature_columns = feature_columns
    )

    recursive_hr_masker = RecursiveHrMasker(0)

    sample_maker = SampleMaker(ts_per_sample, ts_per_sample)

    is_pred_split = NoDiffInitialStatePredictionSplit(ts_per_sample, ts_per_is)

    ts_aggregator = TimeSnippetAggregator(size=frequency_hz*period_s)


    return TransformerPipeline(
        ztransformer,
        feature_label_splitter,
        ts_aggregator,
        meansub,
        sample_maker,
        is_pred_split,
        recursive_hr_masker,
        fft
    )



def get_preprocessing_transformer_ieee(ts_per_sample=30, ts_per_is=2, frequency_hz=32,
                                       period_s=8, donwsampling_ratio = 32/125):
    downsampler = Downsampler(donwsampling_ratio)
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

    ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-BVP-1'], dataset= "ieee_train")

    fft = FFT(BVP_IDX)

    feature_label_splitter = FeatureLabelSplit(
        label_column = "heart_rate",
        feature_columns = feature_columns
    )

    recursive_hr_masker = RecursiveHrMasker(0)

    sample_maker = SampleMaker(ts_per_sample, ts_per_sample)

    is_pred_split = NoDiffInitialStatePredictionSplit(ts_per_sample, ts_per_is)

    ts_aggregator = TimeSnippetAggregator(size=frequency_hz*period_s)


    return TransformerPipeline(
        downsampler,
        ztransformer,
        feature_label_splitter,
        ts_aggregator,
        meansub,
        sample_maker,
        is_pred_split,
        recursive_hr_masker,
        fft
    )
