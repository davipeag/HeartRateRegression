import torch
import numpy as np


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (torch.Tensor(self.x[idx]), torch.tensor([self.y[idx]]).type(torch.FloatTensor))


class DataLoaderFactory():
    
    def __init__(self, transformers, dfs, transformers_ts = None, batch_size_tr=128, batch_size_ts=10**3, dataset_cls=XYDataset):
        self.transformers = transformers
        self.dfs = dfs
        self.dataset_cls = dataset_cls
        self.batch_size_tr = batch_size_tr
        self.batch_size_ts = batch_size_ts
        if transformers_ts is None:
            self.transformers_ts = transformers
        else:
            self.transformers_ts = transformers_ts
    
    def make_set(self, xys, idxs):
        members = [xys[i] for i in idxs]
        return list(map(np.concatenate, zip(*members)))
  
    def make_loader(self, xys, shuffle, batch_size):
        xysn = self.make_set(xys, range(len(xys)))
        ds = self.dataset_cls(*xysn)
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size,
            shuffle=shuffle, num_workers=0)
    
    def make_loaders(self, ts_sub, val_sub):
        idxes = list(range(len(self.dfs)))
        train_idxes = [i for i in idxes if i not in [ts_sub, val_sub]]

        xy_tr = [self.transformers.transform(self.dfs[i]) for i in train_idxes]
        xy_val = [self.transformers_ts.transform(self.dfs[i]) for i in [val_sub]]
        xy_ts = [self.transformers_ts.transform(self.dfs[i]) for i in [ts_sub]]
        
        loader_tr = make_loader(xy_tr, True, self.batch_size_tr)
        loader_val = make_loader(xy_val, False, self.batch_size_ts)
        loader_ts = make_loader(xy_ts, False, self.batch_size_ts)
        
        return loader_tr, loader_val, loader_ts
    