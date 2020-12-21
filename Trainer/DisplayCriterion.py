from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import Trainer
from Trainer import Interfaces
from Trainer.Interfaces import ModelOutput



class CosineSimilarityTripletLoss():
    def __init__(self, margin =1):
        self.margin = margin
        self.cosine = cosine_similarity
        
    
    def __call__(self, output: ModelOutput):
        xa = output.label
        xp = output.prediction[:,0]
        xn = output.prediction[:,1]

        cp = self.cosine(xa, xp)
        cn = self.cosine(xa, xn)
        
        d = cn - cp + self.margin
        
        return np.mean(np.max(np.stack([d, np.zeros(d.shape)], axis= 1), axis=1))
