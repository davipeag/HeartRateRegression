from preprocessing_utils import (
    HZMeanSubstitute, ZTransformer2, FFTXY,
    TimeSnippetAggregator, FeatureLabelSplit,
    TransformerPipeline, Downsampler)


class PreprocessingTransformerGetter():
    def __init__(self):
        self.ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])

    def __call__(self, frequency_hz=32, period_s=8):

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

        ztransformer=self.ztransformer        

        fftxy = FFTXY(BVP_IDX)

        feature_label_splitter = FeatureLabelSplit(
            label_column = "heart_rate",
            feature_columns = feature_columns
        )
        ts_aggregator = TimeSnippetAggregator(size=frequency_hz*period_s)

        return TransformerPipeline(
            ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            fftxy,
            )


class IeeePreprocessingTransformerGetter():
    def __init__(self):
        self.ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-BVP-1'], dataset= "ieee_train")

    def __call__(self, frequency_hz=32, period_s=8, donwsampling_ratio = 32/125):

        downsampler = Downsampler(ratio=donwsampling_ratio)

        feature_columns = [
                    'heart_rate',
                    'wrist-ACC-0',
                    'wrist-ACC-1',
                    'wrist-ACC-2',
                    'wrist-BVP-0',
                    'wrist-BVP-1'
                    ]

        BVP_IDX = [feature_columns.index('wrist-BVP-0'),feature_columns.index('wrist-BVP-1')]
    

        meansub = HZMeanSubstitute()

        ztransformer = self.ztransformer


        fftxy = FFTXY(BVP_IDX)

        feature_label_splitter = FeatureLabelSplit(
            label_column = "heart_rate",
            feature_columns = feature_columns
        )
        ts_aggregator = TimeSnippetAggregator(size=frequency_hz*period_s)

        return TransformerPipeline(
            downsampler,
            ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            fftxy,
        )




#loader_tr, loader_val, loader_ts = make_loaders(ts_sub, val_sub)