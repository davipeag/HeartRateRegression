import torch
import random
import numpy as np

class PceDiscriminatorDataset(torch.utils.data.Dataset):
    def __init__(self, x0, hr0, x1, hr1, label):
        self.x0 = x0
        self.hr0 = hr0
        self.x1 = x1
        self.hr1 = hr1
        self.label = label

    def __len__(self):
        return len(self.x0)
  
    def __getitem__(self, idx):
        return (torch.Tensor(self.x0[idx]), torch.tensor(self.hr0[idx]).type(torch.FloatTensor),
                torch.Tensor(self.x1[idx]), torch.tensor(self.hr1[idx]).type(torch.FloatTensor),
                torch.tensor(self.label[idx]).type(torch.FloatTensor))


class TripletPceDiscriminatorDataset(torch.utils.data.Dataset):
    def __init__(self, xa, hra, xp, hrp, xn, hrn):
        self.xa = xa
        self.hra = hra
        self.xp = xp
        self.hrp = hrp
        self.xn = xn
        self.hrn = hrn

    def __len__(self):
        return len(self.x0)
  
    def __getitem__(self, idx):
        return (torch.Tensor(self.xa[idx]), torch.tensor(self.hra[idx]).type(torch.FloatTensor),
                torch.Tensor(self.xp[idx]), torch.tensor(self.hrp[idx]).type(torch.FloatTensor),
                torch.Tensor(self.xn[idx]), torch.tensor(self.hrn[idx]).type(torch.FloatTensor))

        

class PceDiscriminatorDataLoaderFactory():
    
    def __init__(self, transformers, dfs, transformers_val=None, transformers_ts=None, batch_size_tr=128, batch_size_ts=10**3, dataset_cls=PceDiscriminatorDataset):
        self.transformers = transformers
        self.dfs = dfs
        self.dataset_cls = dataset_cls
        self.batch_size_tr = batch_size_tr
        self.batch_size_ts = batch_size_ts
        if transformers_ts is None:
            self.transformers_ts = transformers
        else:
            self.transformers_ts = transformers_ts
        if transformers_val is None:
            self.transformers_val = transformers
        else:
            self.transformers_val = transformers_val
    
    def process_data(self, dfs, transformer):
      processed = list()
      for i in range(len(dfs)-1):
        df = dfs[i] 
        odf = random.choice(dfs)
        processed.append(transformer.transform(df, odf))
      return list(map(np.concatenate, zip(*processed)))
    
 
    def make_loader(self, dfs, shuffle, batch_size, transformer):
        xysn = self.process_data(dfs, transformer)
        ds = self.dataset_cls(*xysn)
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size,
            shuffle=shuffle, num_workers=0)
    
    def make_loaders(self, ts_sub, val_sub):
        idxes = list(range(len(self.dfs)))
        dfs_tr = [self.dfs[i] for i in idxes if i not in [ts_sub, val_sub]]
        dfs_val = [self.dfs[val_sub], self.dfs[ts_sub]]
        dfs_ts = [self.dfs[ts_sub], self.dfs[val_sub]]
        
        loader_tr = self.make_loader(dfs_tr, True, self.batch_size_tr, self.transformers)
        loader_val = self.make_loader(dfs_val, False, self.batch_size_ts, self.transformers_val)
        loader_ts = self.make_loader(dfs_ts, False, self.batch_size_ts, self.transformers_ts)
        
        return loader_tr, loader_val, loader_ts
