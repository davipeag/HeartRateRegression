import numpy as np

class SimpleEnsembles():
    def __init__(self, get_prediction = lambda r: r["predictons"][1],
                 get_label = lambda r: r["predictons"][0]):
        self.get_prediction = get_prediction
        self.get_label = get_label
    
    def compute_ensemble(self, results):
        ps = [self.get_prediction(v).reshape(-1).numpy() for v in results]
        s = ps[0]
        for p in ps[1:]:
            s = s + p
        return s/len(ps)
    
    def compute_labels(self, results):
        ys = [self.get_label.reshape(-1).numpy() for v in results]
        for i in range(1, len(ys)-1):
            assert np.all(ys[i] == ys[i-1])
        return ys[0]
    

