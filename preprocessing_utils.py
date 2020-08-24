#%%

import numpy as np
import pandas as pd
import os
import time
import copy
from numba import jit


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion



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


def extract_samples(x,y, size=100):
  xs, ys = [], []
  for i in np.arange(0, x.shape[0], size)[:-1]:
    xs.append(x[i:i+size])
    ys.append(y[i: i+ size])

  return np.stack(xs), np.stack(ys)


class LocalMeanReplacer( BaseEstimator, TransformerMixin ):
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


class ZTransformer( BaseEstimator, TransformerMixin ):
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
  
  def fit(self, df, y=None):
    arr = df[self.column_names].to_numpy()
    self.transformer.fit(arr)
    return self
  
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


class ImputeZero( BaseEstimator, TransformerMixin ):
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


class ActivityIdRelabeler(BaseEstimator, TransformerMixin):
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


class Downsampler(BaseEstimator, TransformerMixin):
    
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


class TransformerPipeline():
  def __init__(self, *transformers):
  
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


class FeatureLabelSplit(BaseEstimator, TransformerMixin):
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
           

class TimeSnippetAggregator(BaseEstimator, TransformerMixin):
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


class RemoveLabels(BaseEstimator, TransformerMixin):
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


class LinearImputation(BaseEstimator, TransformerMixin):
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

class FakeNormalizeDZ():
  def __init__(self):
    pass

  def fit(self, x, y=None):
    return self
  
  def transform(self, xy):
    return xy
    
  def reverse_transform(self, xy):
    return xy


class SampleMaker(BaseEstimator, TransformerMixin):
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


class LabelCumSum(BaseEstimator, TransformerMixin):
    def __init__(self):
      pass  

    def fit(self, df, y=None):
      return self

    def transform(self, xy):
      x,y = xy
      return x, np.cumsum(y, axis=1)


class InitialStatePredictionSplit(BaseEstimator, TransformerMixin):
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


class RecursiveHrMasker(BaseEstimator, TransformerMixin):
    def __init__(self, mask_constant=0):
      self.mask_value = mask_constant

    def fit(self, df, y=None):
      return self

    def transform(self, xy):
      xi,yi,xr,yr = xy
      xr[:,:,0,:] = self.mask_value
      return xi,yi,xr,yr


class OurConvLstmToCnnImuFormat(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, *arg, **kargs):
    return self
  
  def transform(self, xy):
      xi,yi,xr,yr = xy
      x = np.concatenate([xi,xr], axis=1) 
      x = np.concatenate([x[:, i:i+1] for i in range(x.shape[1])],axis=3)
      x = x.transpose(0,1,3,2)
      return x, yr
    

class OurConvLstmToAttentionFormat(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, *arg, **kargs):
    return self
  
  def transform(self, xy):
      xi,yi,xr,yr = xy
      x = np.concatenate([xi,xr], axis=1) 
      return x, yr
    
  
class IdentityTransformer(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass

  def fit(self, *arg, **kargs):
    return self
  
  def transform(self, xy):
      return xy
    

class SlidingWindow():
    def __init__(self,
        length,
        step):
      self.length = length
      self.step = step

    def fit(self, df, y=None):
      return self

    def transform(self, xy, y=None):
      x,y = xy
      st = np.arange(0, x.shape[0], self.step)
      ed = st + self.length
      ed = ed[ed<=x.shape[0]]
      st =st[:len(ed)]
      
      xs = list()
      ys = list()
      for s,e in zip(st,ed):
        xs.append(x[s:e])
        ys.append(y[s:e])
      return np.stack(xs), np.stack(ys).reshape(-1, self.length, 1)

class SlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self,
        length,
        step):
      self.length = length
      self.step = step

    def fit(self, df, y=None):
      return self

    def transform(self, xy, y=None):
      x,y = xy
      st = np.arange(0, x.shape[0], self.step)
      ed = st + self.length
      ed = ed[ed<=x.shape[0]]
      st =st[:len(ed)]
      
      xs = list()
      ys = list()
      for s,e in zip(st,ed):
        xs.append(x[s:e])
        ys.append(y[s:e])
      return np.stack(xs), np.stack(ys).reshape(-1, self.length, 1)

class FeatureMeanSubstitute():
  def __init__(self):
    pass

  def fit(self, x, y=None):
    return self
  
  def transform(self, xy):
    x,y = xy

    return np.nanmean(x.swapaxes(3,4),axis=4).reshape(*x.shape[0:2], -1), y
  
class OffsetLabel():
  def __init__(self):
    pass

  def fit(self, x, y=None):
    return self
  
  def transform(self, xy):
    x,y = xy
    return x[0:-1], y[1:]