from preprocessing_utils import (
    HZMeanSubstitute, ZTransformer2, FFTXY, FFTXY_KEEP, FFTXY2,
    TimeSnippetAggregator, FeatureLabelSplit,
    TransformerPipeline, Downsampler, SlidingWindow, IdentityTransformer)


class PreprocessingTransformerGetter():
    def __init__(self):
        self.ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-EDA-0', 'wrist-TEMP-0', 'chest-ACC-0',
                'chest-ACC-1', 'chest-ACC-2', 'chest-Resp-0'])

    def __call__(self, frequency_hz=32, period_s=8, step_s=2, sequence_length = 1):

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
        ts_aggregator = TimeSnippetAggregator(size=frequency_hz*period_s, step=frequency_hz*step_s)
        if sequence_length > 1:
            sequence_step = sequence_length//2
            sliding_window = SlidingWindow(sequence_length, sequence_step)
        else:
            sliding_window = IdentityTransformer()

        return TransformerPipeline(
            ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            fftxy,
            sliding_window
            )


class IeeePreprocessingTransformerGetter():
    def __init__(self, donwsampling_ratio=32/125):
        self.ztransformer = ZTransformer2(['heart_rate', 'wrist-ACC-0', 'wrist-ACC-1', 'wrist-ACC-2',
                'wrist-BVP-0', 'wrist-BVP-1'], dataset= "ieee_train")
        self.downsampling_ratio = donwsampling_ratio
        self.downsampler = Downsampler(donwsampling_ratio)
    
    def set_downsampling_ratio(self, ratio):
        self.downsampler = Downsampler(ratio)
        self.downsampling_ratio = ratio


    def __call__(self, frequency_hz=125, period_s=8, step_s = 2, sequence_length=1):

        
        feature_columns = [
                    'heart_rate',
                    'wrist-ACC-0',
                    'wrist-ACC-1',
                    'wrist-ACC-2',
                    'wrist-BVP-0',
                    'wrist-BVP-1'
                    ]

        BVP_IDX = [feature_columns.index('wrist-BVP-0'),feature_columns.index('wrist-BVP-1')]
        # FFT_IDXS = list(range(1,len(feature_columns)))
        FFT_IDXS = BVP_IDX

        frequency = frequency_hz*self.downsampling_ratio

        meansub = HZMeanSubstitute()
        
        fftxy = FFTXY2(FFT_IDXS)

        feature_label_splitter = FeatureLabelSplit(
            label_column = "heart_rate",
            feature_columns = feature_columns
        )

        ts_aggregator = TimeSnippetAggregator(size=int(frequency*period_s), step=int(frequency*step_s))

        if sequence_length > 1:
            sequence_step = sequence_length//2
            sliding_window = SlidingWindow(sequence_length, sequence_step)
        else:
            sliding_window = IdentityTransformer()

        return TransformerPipeline(
            self.downsampler,
            self.ztransformer,
            feature_label_splitter,
            ts_aggregator,
            meansub,
            fftxy,
            sliding_window
        )





#loader_tr, loader_val, loader_ts = make_loaders(ts_sub, val_sub)