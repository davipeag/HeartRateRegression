import copy
import numpy as np
import heartpy as hp


def rescale(x, minv, scale=0.5):
    x = copy.deepcopy(x)
    x[np.abs(x) < minv] = x[np.abs(x) < minv]*scale
    return x


def sliding_window(x, window_size, window_step):
    st = np.arange(0, x.shape[0], window_step)
    ed = st + window_size
    ed = ed[ed <= x.shape[0]]
    st = st[:len(ed)]
    xs = list()
    for s, e in zip(st, ed):
        xs.append(x[int(s):int(e)])
    return np.stack(xs)


def low_diff_idxes(x, max_diff):
    return np.insert(np.abs(x[1:] - x[:-1]) < max_diff, 0, True)


def moving_average(a, n=3):
    h = (n-1)//2
    st = n-1 - h
    ed = len(a) - h
    nv = np.full(a.shape, 0)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    nv[st:ed] = ret[n - 1:] / n
    nv[:st] = nv[st]
    nv[ed:] = nv[ed-1]
    return nv


def fore_substitute(vals, to_sub):
    vals = copy.deepcopy(vals)
    fore = vals[0]
    for i, s in enumerate(to_sub):
        if s:
            vals[i] = fore
        else:
            fore = vals[i]
    return vals


def nan_substitute(vals, to_sub):
    vals = copy.deepcopy(vals)
    vals[to_sub] = np.nan
    return vals


def nan_low_diff_idxes(vals, max_diff):
    valid = np.full(vals.shape, False, bool)
    fore = vals[0]
    for i, s in enumerate(vals):
        if np.abs(s-fore) < max_diff:
            valid[i] = True
        fore = s
    return valid


def linear_imputation(arr):
    i = np.where(~np.isnan(arr))[0]
    ib = i[:-1]
    ie = i[1:]

    for b, e in zip(ib, ie):
        d = e-b
        ve = arr[e]
        vb = arr[b]

        arr[b+1:e] = (np.arange(d-1, 0, -1)*vb + np.arange(1, d)*ve)/d

    arr[e+1:] = arr[e]
    arr[:i[0]] = arr[i[0]]
    return arr


class GetBPM():
    def __init__(self, cutoff, sample_rate=700):
        self.cutoff = cutoff
        self.sample_rate = sample_rate

    def __call__(self, signal):
        s = signal
        sample_rate = self.sample_rate
        s = hp.remove_baseline_wander(s, sample_rate)
        s = hp.filter_signal(s, cutoff=self.cutoff,
                             sample_rate=sample_rate, filtertype='notch')
        if not (self.cutoff == 0.5):
            s = hp.scale_data(s, sample_rate)
        try:
            wd, m = hp.process(s, sample_rate)
            v = m['bpm']
            if (v > 35) and (v < 210):
                return v
            else:
                return np.nan
        except Exception:
            return np.nan


def multi_try_bpm(s, processors):
    for proc in processors:
        v = proc(s)
        if not np.isnan(v):
            return v
    return np.nan


def get_bpm(x, sample_rate):
    return multi_try_bpm(
        s, [GetBPM(0.01, sample_rate),
            GetBPM(0.5, sample_rate),
            GetBPM(0.05, sample_rate),
            GetBPM(0.1, sample_rate)])
    # try:
    #     return hp.process(x, sample_rate, bpmmin=35, bpmmax=210)[1]['bpm']
    # except Exception:
    #     return np.nan


def wesad_ecg_preprocessing(ecg, sample_rate=700):
    d = ecg
    # d = rescale(d, 0.8, 0.8)
    # d = hp.remove_baseline_wander(d, sample_rate)

    # d = hp.scale_data(d, sample_rate)
    # d = hp.filter_signal(
    #     d, cutoff=0.01, sample_rate=sample_rate, filtertype='notch')
    c = np.array([get_bpm(v, sample_rate)
                  for v in sliding_window(d, sample_rate*8, sample_rate*2)])
    max_diff = 10

    i1 = low_diff_idxes(c, max_diff) &  low_diff_idxes(np.flip(c), max_diff) & (c > 35) & (c < 210)

    c2 = (nan_substitute(c, ~i1) + nan_substitute(np.flip(c), ~i1))/2

    c2 = (fore_substitute(c, ~i1) + fore_substitute(np.flip(c), ~i1))/2

    i2 = nan_low_diff_idxes(c2, max_diff*3/4) &  nan_low_diff_idxes(np.flip(c2), max_diff*3/4)

    c3 = (nan_substitute(c2, ~i2) + nan_substitute(np.flip(c2), ~i2))/2

    c3 = (fore_substitute(c2, ~i2) + fore_substitute(np.flip(c2), ~i2))/2

    i3 = nan_low_diff_idxes(c3, max_diff*2/3) &  nan_low_diff_idxes(np.flip(c3), max_diff*2/3)

    i = i1 & i2 & i3    
    # i1 = low_diff_idxes(c, 20) & low_diff_idxes(
    #     np.flip(c), 20) & (c > 35) & (c < 210)
    # c2 = (nan_substitute(c, ~i1) + nan_substitute(np.flip(c), ~i1))/2
    # i2 = nan_low_diff_idxes(c2, 20) & nan_low_diff_idxes(np.flip(c2), 20)

    # i = i1 & i2

    return linear_imputation(nan_substitute(c, ~i))
