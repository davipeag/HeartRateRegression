#%%
import numpy as np
from numba import jit
import torch
from torch import nn
import pandas as pd
from sklearn import preprocessing
import copy

@jit(nopython=True)
def linear_imputation(arr):
  i = np.where(~np.isnan(arr))[0]
  ib = i[:-1]
  ie = i[1:]

  for b,e  in zip(ib, ie):
    d = e-b
    ve = arr[e]
    vb = arr[b]

    arr[b+1:e] = (np.arange(d-1,0,-1)*vb + np.arange(1, d)*ve)/d

  arr[e+1:] = arr[e]
  arr[:i[0]] = arr[i[0]]
  return arr

@jit(nopython=True)
def local_mean_fill(x, mrange):
  for icol in range(x.shape[1]):
    attr = x[:,icol]
    idxs = np.where(np.isnan(attr))[0]
    for i in idxs:
      ilower = i-mrange
      iupper = i+mrange
      x[i,icol] = np.nanmean(attr[ilower:iupper])
  return x

def extract_samples(x,y, size=100):
  xs, ys = [], []
  for i in np.arange(0, x.shape[0], size)[:-1]:
    xs.append(x[i:i+size])
    ys.append(y[i: i+ size])

  return np.stack(xs), np.stack(ys)


class LinearImputation():
    def __init__(self, column = "heart_rate"):
      self.column = column 

    def fit(self, df, y=None):
      return self

    def transform(self, df, y=None):
      arr = df[self.column].to_numpy()
      df[self.column] = linear_imputation(arr)
      return df
      

class HZMeanSubstitute():
  def __init__(self):
    pass

  def fit(self, x, y=None):
    return self
  
  def transform(self, xy):
    x,y = xy  
    inds = list()
    hzs = list()
    for ind in x:
      hz = np.nanmean(ind[:,:, 0])
      ind[:,:, 0] = hz
      hzs.append(hz)
      inds.append(ind)
    nx,ny = np.stack(inds), np.array(hzs)
    #print(nx.shape, ny.shape)
    return nx, ny


class DeltaHzToLabel():
  def __init__(self):
    pass

  def fit(self, x, y=None):
    return self
  
  def transform(self, xy):
    x,y = xy
    yn = y[1:] - y[:-1]
    xn = x[:-1]  
    return xn, yn


class NormalizeDZ():
  def __init__(self):
    pass

  def fit(self, x, y=None):
    return self
  
  def transform(self, xy):
    x,y  = xy
    return x, (y-0.000737)/0.065
  
  def reverse_transform(self, xy):
    x,y = xy
    return x, (y*0.065)+0.000737


class SampleMaker():
    def __init__(self,
                 count,
                 step=1
                 ):
      self.count = count
      self.step=step  

    def fit(self, df, y=None):
      return self

    def transform(self, xy):
      x,y = xy
      idxs = np.arange(0, len(x)-self.count, self.step)
      yn = y.reshape(-1,1)
      xs = list()
      ys = list()
      for i in idxs:
        xs.append(np.concatenate(x[i:i+self.count], axis=1))
        ys.append(np.concatenate(yn[i:i+self.count]))
      return np.stack(xs), np.stack(ys)


class LabelCumSum():
    def __init__(self):
      pass  

    def fit(self, df, y=None):
      return self

    def transform(self, xy):
      x,y = xy
      return x, np.cumsum(y, axis=1)


class InitialStatePredictionSplit():
    def __init__(self,
                 total_window,
                 initial_window):
      self.total_window = total_window
      self.initial_window = initial_window

    def fit(self, df, y=None):
      return self

    def transform(self, xy):
      x,y = xy
      xn = x.reshape([x.shape[0], self.total_window, x.shape[2]//self.total_window, x.shape[3]])
      yn = y.reshape([y.shape[0], self.total_window, y.shape[1]//self.total_window])
      prediction_window = self.total_window - self.initial_window
      xi,yi = xn[:,:self.initial_window, :, :], xn[:,:self.initial_window, :, 0:1]
      yp = yn[:,self.initial_window:] - yn[:,self.initial_window-1:self.initial_window] 
      xp = xn[:, self.initial_window:, :, :]
      return np.swapaxes(xi, 2,3), np.mean(yi, axis=2), np.swapaxes(xp, 2,3), yp


class RecursiveHrMasker():
    def __init__(self, mask_constant=0):
      self.mask_value = mask_constant

    def fit(self, df, y=None):
      return self

    def transform(self, xy):
      xi,yi,xr,yr = xy
      xr[:,:,0,:] = self.mask_value
      return xi,yi,xr,yr


class LocalMeanReplacer():
  def __init__(self, mean_width = 20,
               column_names = [
              'h_temperature', 
              'h_xacc16','h_yacc16', 'h_zacc16',
              'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr',
              'h_xmag', 'h_ymag', 'h_zmag',
              'c_temperature',
              'c_xacc16', 'c_yacc16', 'c_zacc16', 'c_xacc6', 'c_yacc6', 'c_zacc6',
              'c_xgyr', 'c_ygyr', 'c_zgyr',
              'c_xmag', 'c_ymag', 'c_zmag',
              'a_temperature',
              'a_xacc16', 'a_yacc16', 'a_zacc16', 'a_xacc6',
              'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr',
              'a_xmag', 'a_ymag','a_zmag'
        ]):
    self.mrange = mean_width
    self.column_names = column_names
  
  def fit(self, x, y=None):
    return self
  
  def transform(self, df, y=None):
    df = copy.deepcopy(df)
    arr = df[self.column_names].to_numpy()
    arr = local_mean_fill(arr, self.mrange)
    df[self.column_names] = arr
    return df
  
  def inverse_transform(self, df, y=None):
    return df


class ZTransformer():
  def __init__(self,
               column_names = [
              'heart_rate', 'h_temperature', 
              'h_xacc16','h_yacc16', 'h_zacc16',
              'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr',
              'h_xmag', 'h_ymag', 'h_zmag',
              'c_temperature',
              'c_xacc16', 'c_yacc16', 'c_zacc16', 'c_xacc6', 'c_yacc6', 'c_zacc6',
              'c_xgyr', 'c_ygyr', 'c_zgyr',
              'c_xmag', 'c_ymag', 'c_zmag',
              'a_temperature',
              'a_xacc16', 'a_yacc16', 'a_zacc16', 'a_xacc6',
              'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr',
              'a_xmag', 'a_ymag','a_zmag'
        ]):
    self.column_names = column_names    
    self.transformer = preprocessing.StandardScaler()
    self.transformer.mean_ = np.array([ 1.09872508e+02,  3.26525787e+01, -4.96078635e+00,  3.58775813e+00,
            3.16841666e+00, -4.88941959e+00,  3.58426743e+00,  3.34947892e+00,
          -3.70474442e-03,  3.21353993e-02, -6.16753052e-03,  2.11685710e+01,
          -1.43890823e+01, -2.16445082e+01,  3.61497439e+01,  3.76375451e-01,
            8.51220461e+00, -1.51812923e+00,  2.44862418e-01,  8.50599843e+00,
          -1.18356330e+00,  5.53869793e-03,  8.37116240e-03, -2.10622194e-02,
            3.57970919e+00, -3.18577260e+01,  5.51682811e+00,  3.36851338e+01,
            9.54544754e+00, -1.23247637e-01, -2.46150312e+00,  9.52134760e+00,
          -2.06264580e-02, -2.03983029e+00,  8.63514327e-03, -3.45012192e-02,
            7.75203048e-03, -3.27210215e+01,  1.59330413e+00,  1.68904416e+01])

    self.transformer.var_ = np.array([6.69256204e+02, 3.40134569e+00, 3.58205561e+01, 3.94112412e+01,
          1.47757355e+01, 3.59127521e+01, 3.66720917e+01, 1.47505850e+01,
          1.68322905e+00, 7.86800278e-01, 2.10927713e+00, 5.77027950e+02,
          5.80540648e+02, 4.25003650e+02, 2.51402668e+00, 2.62612219e+00,
          1.79346885e+01, 1.74220212e+01, 2.62271089e+00, 1.76966554e+01,
          1.75406500e+01, 1.61626949e-01, 2.90447973e-01, 8.55228249e-02,
          2.70660934e+02, 2.59825372e+02, 3.96884575e+02, 1.40271582e+00,
          3.26223058e+01, 4.67766972e+01, 1.26522303e+01, 2.85837577e+01,
          4.07323877e+01, 1.01834375e+01, 1.15252135e+00, 3.55934492e-01,
          3.39499680e+00, 3.56401566e+02, 4.67070326e+02, 4.12438440e+02])

    self.transformer.scale_ = np.array([25.86998654,  1.84427376,  5.98502766,  6.2778373 ,  3.84392188,
            5.99272493,  6.05574865,  3.84064903,  1.29739317,  0.88701763,
            1.45233506, 24.02140607, 24.09441114, 20.61561665,  1.58556825,
            1.62053145,  4.23493666,  4.17396948,  1.61947859,  4.20673929,
            4.18815591,  0.40202854,  0.53893225,  0.29244286, 16.451776  ,
          16.1190996 , 19.92196213,  1.18436304,  5.71159398,  6.83934918,
            3.55699737,  5.346378  ,  6.38219302,  3.19114986,  1.07355547,
            0.59660246,  1.84255171, 18.87860076, 21.61180986, 20.30858045])
  
  def fit(self, df, y=None):
    return self
    # arr = df[self.column_names].to_numpy()
    # self.transformer.fit(arr)
    # return self
  
  def transform(self, df, y=None):
    df = copy.deepcopy(df)
    arr = df[self.column_names].to_numpy()
    arr = self.transformer.transform(arr)
    df[self.column_names] = arr
    return df
  
  def inverse_transform(self, df, y=None):
    arr = df[self.column_names].to_numpy()
    arr = self.transformer.inverse_transform(arr)
    df[self.column_names] = arr
    return df


class ImputeZero():
  def __init__(self, column_name = "heart_rate"):
    self.column_name = column_name    
    
  def fit(self, df, y=None):
    return self
  
  def transform(self, df, y=None):
    df = copy.deepcopy(df)
    arr = df[self.column_name].to_numpy()
    data_no_NaN = np.zeros(arr.shape)
    data_no_NaN[~np.isnan(arr)] = arr[~np.isnan(arr)]
    df[self.column_name] = data_no_NaN
    return df
    
  def inverse_transform(self, df, y=None):
    return df


class TimeSnippetAggregator():
    def __init__(self,
        size=100,
        label_collapser_function = lambda v: np.median(v, axis=1)):
      self.size = size
      self.label_collapser_function = label_collapser_function  
    
    def fit(self, df, y=None):
      return self

    def transform(self, xydfs, y=None):
      x,y = map(lambda v: v.to_numpy(), xydfs)
      sx, sy = extract_samples(x,y,self.size)

      return (np.expand_dims(sx, axis=1), self.label_collapser_function(sy))


class ActivityIdRelabeler():
    def __init__(self,
                 column_name = "activity_id",
                 labels = [ 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24 ]):
      
      self.column_name = column_name
      self.label_mapping = dict()

      for i,l in enumerate(labels):
        self.label_mapping[l]=i
    
    def fit(self, df, y=None):
      return self

    def transform(self, df, y=None):
      df = copy.deepcopy(df)
      arr = df[self.column_name]
      aids = np.array(list(map(lambda x: self.label_mapping[x],
                                         df[self.column_name])))
      df[self.column_name] = aids     
      return df


class Downsampler():
    
    def __init__(self, ratio=0.3, init_idx=0):
      self.ratio = ratio
      self.init_idx = 0

    
    def fit(self, df, y=None):
      return self

    def transform(self, df, y=None):
      df = copy.deepcopy(df)
      idx = np.arange(self.init_idx, df.shape[0], int(1/self.ratio) )
      arr = df.to_numpy()
      darr = arr[idx]
      ddf = pd.DataFrame(darr, columns=df.columns)     
      
      return ddf


class FeatureLabelSplit():
    def __init__(
        self,
        feature_columns= [
          'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
          'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
          'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
          'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
          'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
          'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
          'a_ymag', 'a_zmag'],
        label_column="activity_id"):
      self.feature_columns = feature_columns
      self.label_column = label_column
    
    def fit(self, df, y=None):
      return self

    def transform(self, df, y=None):
      return (df[self.feature_columns], df[self.label_column])


class RemoveLabels():
    def __init__(self,
        labels_to_remove=[0]):
      self.labels_to_remove = labels_to_remove

    def fit(self, df, y=None):
      return self

    def transform(self, xy, y=None):
      x,y = xy
      idxs = np.full(len(y), True)
      for lab in self.labels_to_remove:
        idxs = (idxs & (y != lab))
      return x[idxs], y[idxs]


class TransformerPipeline():
  def __init__(self,
               *transformers):
  
    self.transformers = transformers

  def fit(self, df, y=None):
    for transformer in self.transformers:
      transformer.fit(df)
    return self
  
  def transform(self, df, y=None):
    df = self.transformers[0].transform(df)
    for transformer in self.transformers[1:]:
      df = transformer.transform(df)
    return df 

  
  def inverse_transform(self, df, y=None):
    df = self.transformers[-1].transform(df)
    for transformer in list(reversed(self.transformers))[1:]:
      df = transformer.transform(df)
    return df 


recursive_hr_masker = RecursiveHrMasker(0)

label_cum_sum = LabelCumSum()

hr_lin_imputation = LinearImputation("heart_rate")

meansub = HZMeanSubstitute()

deltahztolabel = DeltaHzToLabel()

normdz = NormalizeDZ()

local_mean_imputer = LocalMeanReplacer()

ztransformer = ZTransformer()

zero_imputer = ImputeZero()

activity_id_relabeler = ActivityIdRelabeler()

downsampler = Downsampler(1)

feature_label_splitter = FeatureLabelSplit(
    feature_columns = [
          'heart_rate', 'h_temperature', 'h_xacc16', 'h_yacc16', 'h_zacc16',
          'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr', 'h_xmag',
          'h_ymag', 'h_zmag', 'c_temperature', 'c_xacc16', 'c_yacc16', 'c_zacc16',
          'c_xacc6', 'c_yacc6', 'c_zacc6', 'c_xgyr', 'c_ygyr', 'c_zgyr', 'c_xmag',
          'c_ymag', 'c_zmag', 'a_temperature', 'a_xacc16', 'a_yacc16', 'a_zacc16',
          'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr', 'a_xmag',
          'a_ymag', 'a_zmag']
)
ts_aggregator = TimeSnippetAggregator(size=300)
label_remover = RemoveLabels([0])

class OurConvLSTMTransformers():
    def __init__(self, transformers, transformers_ts, normdz, ztransformer):
        self.transformers = transformers
        self.transformers_ts = transformers_ts
        self.normdz = normdz
        self.ztransformer = ztransformer


def get_transformers(ts_per_sample=162, ts_per_is=2):

    sample_maker = SampleMaker(ts_per_sample, ts_per_sample//2)

    sample_maker_ts = SampleMaker(ts_per_sample, ts_per_sample)

    is_pred_split = InitialStatePredictionSplit(ts_per_sample, ts_per_is)

    transformers = TransformerPipeline(ztransformer, hr_lin_imputation, local_mean_imputer,
                                    activity_id_relabeler, downsampler,
                                    feature_label_splitter,
                                    ts_aggregator, meansub, deltahztolabel, normdz,
                                    sample_maker, label_cum_sum, is_pred_split, recursive_hr_masker)

    transformers_ts = TransformerPipeline(ztransformer, hr_lin_imputation, local_mean_imputer,
                                    activity_id_relabeler,downsampler,
                                    feature_label_splitter,
                                    ts_aggregator, meansub, deltahztolabel, normdz,
                                    sample_maker_ts, label_cum_sum, is_pred_split,recursive_hr_masker)

    return OurConvLSTMTransformers(transformers, transformers_ts, normdz, ztransformer)

#preprocessor = get_transformers()
# #transformers.fit(df_full_train)
# # xys_protocol_t = [transformers.transform(df) for df in dfs_train[:-1]]
# #xys_optional_t= [transformers.transform(df) for df in dfs_test]

# val_sub = 7

# train_ts = [0,1,2,3,5,6,7] 

# val_idx = train_ts.index(val_sub)
# train_idxes = train_ts[:val_idx] + train_ts[val_idx+1:] 

# xy_tr = [transformers.transform(dfs_train[i]) for i in train_idxes]
# xy_val = [transformers_ts.transform(df) for df in dfs_train[4:5]]
# xy_ts = [transformers_ts.transform(dfs_train[i]) for i in [val_sub]]
# #xys_optional_t= [transformers.transform(df) for df in dfs_test]
# print(train_idxes)
# print(val_sub)
