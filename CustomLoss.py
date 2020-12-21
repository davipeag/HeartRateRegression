import torch


class CosineSimilarityTripletLoss():
    def __init__(self, margin =1):
        self.cosine = torch.nn.CosineSimilarity()
        self.margin = margin
        self.triplet_loss = torch.nn.TripletMarginLoss(margin)
    
    def to(self, device):
        self.cosine.to(device)
        self.triplet_loss.to(device)
    
    def __call__(self, xa, xp, xn):
        cp = self.cosine(xa, xp).unsqueeze(1)
        cn = self.cosine(xa, xn).unsqueeze(1)

        # d = cp - cn + self.margin
        dp = 1- cp
        dn = 1-cn
        return self.triplet_loss(dp*0, dp, dn)