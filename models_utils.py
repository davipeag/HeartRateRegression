import torch 
import numpy as np 
import random
import torch 
import numpy as np 
import random

def make_set(xys, idxs):
  members = [xys[i] for i in idxs]
  return list(map(np.concatenate, zip(*members)))
  

def one_subject_out_split(xys, train_idxs=[0,1,2,3,6,7],
                          validation_idxs=[4],
                          test_idxs=[5]):
  return [make_set(xys, idxs) for idxs in [train_idxs, validation_idxs,
                                           test_idxs]
         ]
  

class OurConvLstmDataset(torch.utils.data.Dataset):
  def __init__(self,
               xis, yis, xrs, yrs):
    self.xis = xis
    self.yis = yis
    self.xrs = xrs
    self.yrs = yrs

  def __len__(self):
    return len(self.yis)
  
  def __getitem__(self, idx):
    return (torch.Tensor(self.xis[idx]), torch.tensor(self.yis[idx]).type(torch.FloatTensor),
            torch.Tensor(self.xrs[idx]), torch.tensor(self.yrs[idx]).type(torch.FloatTensor))


class DatasetXY(torch.utils.data.Dataset):
  def __init__(self, x,y):
    self.x = x
    self.y = y
  def __len__(self):
    return len(self.x)
  
  def __getitem__(self, idx):
    return torch.FloatTensor(self.x[idx]), torch.FloatTensor(self.y[idx])
            


def make_loader(xys, dataset_cls, batch_size, shuffle = True, num_workers=0):
  xysn = make_set(xys, range(len(xys)))
  ds = dataset_cls(*xysn)
  return torch.utils.data.DataLoader(
  ds, batch_size=batch_size,
    shuffle=shuffle, num_workers=num_workers)

def reset_seeds(seed=1234):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp






