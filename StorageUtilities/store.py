import os
import pickle

class MakePath():
    def __init__(self, base:str):
        self.base = base
    
    def __call__(self, relative: str):
        return os.path.join(self.base, relative)


class PickleStore():
    def __init__(self, base_dir):
        self.base = base_dir
    
    def path(self, key):
        return os.path.join(self.base, key)
    
    def dump(self, obj, key):
        path = self.path(key)
        with open(path, "w") as f:
            pickle.dump(obj, f)
    
    def exists(self, key):
        return os.path.isfile(self.path(key))
    
    def load(self, key):
        if self.exists(key):
            with open(self.path(key), "r") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"no value stored in {key}")
