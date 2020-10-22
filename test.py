
#%%

samps = [
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0, 'lr': 0.001, 'lin_dropout': 0.25, 'lin_size': 64, 'nlin_layers': 1, 'feedforward_expansion': 1, 'nhead': 4, 'ndec_layers': 1, 'nenc_layers': 1, 'conv_dropout': 0, 'nconv_layers': 2, 'conv_filters': 128, 'nfeatures': 4}, 9.696053],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0.25, 'lin_size': 32, 'nlin_layers': 2, 'feedforward_expansion': 4, 'nhead': 1, 'ndec_layers': 4, 'nenc_layers': 2, 'conv_dropout': 0, 'nconv_layers': 4, 'conv_filters': 128, 'nfeatures': 4}, 8.113631],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0, 'lin_size': 16, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 4, 'ndec_layers': 1, 'nenc_layers': 2, 'conv_dropout': 0, 'nconv_layers': 0, 'conv_filters': 128, 'nfeatures': 4}, 12.264346], 
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0.25, 'lin_size': 16, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 4, 'ndec_layers': 1, 'nenc_layers': 4, 'conv_dropout': 0, 'nconv_layers': 2, 'conv_filters': 32, 'nfeatures': 4}, 9.856819],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0.25, 'lin_size': 64, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 2, 'ndec_layers': 2, 'nenc_layers': 4, 'conv_dropout': 0, 'nconv_layers': 2, 'conv_filters': 32, 'nfeatures': 4}, 8.00527],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0.25, 'lin_size': 32, 'nlin_layers': 1, 'feedforward_expansion': 1, 'nhead': 2, 'ndec_layers': 2, 'nenc_layers': 1, 'conv_dropout': 0, 'nconv_layers': 2, 'conv_filters': 64, 'nfeatures': 4}, 9.354283],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0, 'lin_size': 16, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 1, 'ndec_layers': 4, 'nenc_layers': 2, 'conv_dropout': 0.25, 'nconv_layers': 4, 'conv_filters': 32, 'nfeatures': 4},9.876608],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0.25, 'lin_size': 16, 'nlin_layers': 1, 'feedforward_expansion': 2, 'nhead': 4, 'ndec_layers': 1, 'nenc_layers': 2, 'conv_dropout': 0.5, 'nconv_layers': 0, 'conv_filters': 64, 'nfeatures': 4}, 10.251151],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0.5, 'lin_size': 32, 'nlin_layers': 1, 'feedforward_expansion': 2, 'nhead': 1, 'ndec_layers': 4, 'nenc_layers': 1, 'conv_dropout': 0.25, 'nconv_layers': 4, 'conv_filters': 128, 'nfeatures': 4}, 13.287586],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.001, 'lin_dropout': 0, 'lin_size': 16, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 4, 'ndec_layers': 1, 'nenc_layers': 1, 'conv_dropout': 0.5, 'nconv_layers': 0, 'conv_filters': 32, 'nfeatures': 4}, 10.717534],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0, 'lr': 0.001, 'lin_dropout': 0.25, 'lin_size': 32, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 2, 'ndec_layers': 4, 'nenc_layers': 1, 'conv_dropout': 0.25, 'nconv_layers': 4, 'conv_filters': 64, 'nfeatures': 4}, 8.276257],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0.5, 'lin_size': 32, 'nlin_layers': 1, 'feedforward_expansion': 2, 'nhead': 2, 'ndec_layers': 4, 'nenc_layers': 1, 'conv_dropout': 0.5, 'nconv_layers': 2, 'conv_filters': 128, 'nfeatures': 4}, 9.567037],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0.5, 'lin_size': 32, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 2, 'ndec_layers': 1, 'nenc_layers': 1, 'conv_dropout': 0.25, 'nconv_layers': 2, 'conv_filters': 32, 'nfeatures': 4}, 9.874631],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0.25, 'lin_size': 64, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 1, 'ndec_layers': 2, 'nenc_layers': 4, 'conv_dropout': 0.5, 'nconv_layers': 2, 'conv_filters': 128, 'nfeatures': 4}, 13.809696],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.0001, 'lin_dropout': 0.5, 'lin_size': 64, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 2, 'ndec_layers': 2, 'nenc_layers': 4, 'conv_dropout': 0, 'nconv_layers': 0, 'conv_filters': 64, 'nfeatures': 4}, 11.635924],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.001, 'lin_dropout': 0.25, 'lin_size': 32, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 4, 'ndec_layers': 2, 'nenc_layers': 1, 'conv_dropout': 0.5, 'nconv_layers': 4, 'conv_filters': 128, 'nfeatures': 4}, 18.5427],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.0001, 'lin_dropout': 0.5, 'lin_size': 32, 'nlin_layers': 1, 'feedforward_expansion': 4, 'nhead': 1, 'ndec_layers': 1, 'nenc_layers': 4, 'conv_dropout': 0, 'nconv_layers': 2, 'conv_filters': 32, 'nfeatures': 4},9.2107525],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0.5, 'lin_size': 64, 'nlin_layers': 1, 'feedforward_expansion': 1, 'nhead': 4, 'ndec_layers': 2, 'nenc_layers': 1, 'conv_dropout': 0.5, 'nconv_layers': 2, 'conv_filters': 128, 'nfeatures': 4}, 8.992032],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0.0001, 'lr': 0.0001, 'lin_dropout': 0.5, 'lin_size': 64, 'nlin_layers': 1, 'feedforward_expansion': 1, 'nhead': 4, 'ndec_layers': 1, 'nenc_layers': 1, 'conv_dropout': 0.5, 'nconv_layers': 4, 'conv_filters': 128, 'nfeatures': 4}, 9.770515],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0, 'lin_size': 32, 'nlin_layers': 1, 'feedforward_expansion': 4, 'nhead': 4, 'ndec_layers': 1, 'nenc_layers': 2, 'conv_dropout': 0.5, 'nconv_layers': 2, 'conv_filters': 32, 'nfeatures': 4}, 9.81467],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0, 'lr': 0.001, 'lin_dropout': 0.5, 'lin_size': 64, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 2, 'ndec_layers': 2, 'nenc_layers': 2, 'conv_dropout': 0, 'nconv_layers': 2, 'conv_filters': 32, 'nfeatures': 4}, 8.802119],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0, 'lin_size': 64, 'nlin_layers': 1, 'feedforward_expansion': 4, 'nhead': 1, 'ndec_layers': 1, 'nenc_layers': 1, 'conv_dropout': 0.5, 'nconv_layers': 0, 'conv_filters': 32, 'nfeatures': 4}, 13.801493],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.001, 'lin_dropout': 0.25, 'lin_size': 64, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 2, 'ndec_layers': 1, 'nenc_layers': 2, 'conv_dropout': 0.25, 'nconv_layers': 2, 'conv_filters': 64, 'nfeatures': 4}, 10.684896],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0, 'lin_size': 16, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 4, 'ndec_layers': 4, 'nenc_layers': 4, 'conv_dropout': 0.25, 'nconv_layers': 2, 'conv_filters': 64, 'nfeatures': 4}, 9.678957],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.0001, 'lin_dropout': 0.5, 'lin_size': 32, 'nlin_layers': 2, 'feedforward_expansion': 4, 'nhead': 2, 'ndec_layers': 4, 'nenc_layers': 2, 'conv_dropout': 0.5, 'nconv_layers': 4, 'conv_filters': 64, 'nfeatures': 4}, 9.872826],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0.5, 'lin_size': 16, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 4, 'ndec_layers': 4, 'nenc_layers': 1, 'conv_dropout': 0.25, 'nconv_layers': 2, 'conv_filters': 32, 'nfeatures': 4}, 8.28792],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0.5, 'lin_size': 32, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 4, 'ndec_layers': 4, 'nenc_layers': 2, 'conv_dropout': 0.5, 'nconv_layers': 0, 'conv_filters': 128, 'nfeatures': 4}, 9.811581],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 64, 'weight_decay': 0, 'lr': 0.0001, 'lin_dropout': 0, 'lin_size': 16, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 4, 'ndec_layers': 2, 'nenc_layers': 2, 'conv_dropout': 0, 'nconv_layers': 2, 'conv_filters': 128, 'nfeatures': 4}, 5.674814],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 256, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0.25, 'lin_size': 32, 'nlin_layers': 1, 'feedforward_expansion': 4, 'nhead': 4, 'ndec_layers': 2, 'nenc_layers': 4, 'conv_dropout': 0.25, 'nconv_layers': 2, 'conv_filters': 64, 'nfeatures': 4}, 9.857917],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0.0001, 'lr': 0.0001, 'lin_dropout': 0.25, 'lin_size': 64, 'nlin_layers': 2, 'feedforward_expansion': 2, 'nhead': 2, 'ndec_layers': 4, 'nenc_layers': 2, 'conv_dropout': 0.5, 'nconv_layers': 0, 'conv_filters': 32, 'nfeatures': 4}, 12.066545],
    [{'val_sub': 4, 'ts_sub': 0, 'batch_size': 128, 'weight_decay': 0.0001, 'lr': 0.001, 'lin_dropout': 0.5, 'lin_size': 64, 'nlin_layers': 2, 'feedforward_expansion': 1, 'nhead': 1, 'ndec_layers': 1, 'nenc_layers': 2, 'conv_dropout': 0, 'nconv_layers': 0, 'conv_filters': 64, 'nfeatures': 4}, 10.610379],
]

ssamps = sorted(samps, key=lambda x: x[1])

ssamps[0]
# %%

from collections import defaultdict

def used_ranges(dicts):
    ranges = defaultdict(set)
    for d in dicts:
        for k,v in d.items():
            ranges[k].add(v)
    return ranges

a = used_ranges([s[0] for s in ssamps[0:10]])
b = used_ranges([s[0] for s in ssamps[0:3]])

a, b
#ssamps[19]
# %%
