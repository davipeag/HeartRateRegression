from Trainer.Interfaces import IBatchComputer, ModelOutput
import torch

class BatchComputerXY(IBatchComputer):
    
    def __init__(self, model, criterion, device, name = None):
        self._model = model
        self._criterion = criterion
        self.device = device
        if name is None:
            self._name =  model.__class__.__name__
        else:
            self._name = name

    @property
    def model(self):
        return self._model
    
    @property
    def name(self):
        return self._name
    
    @property
    def criterion(self):
        return self._criterion
    
    def compute_loss(self, batch) -> torch.tensor:
        output = self.compute_batch(batch)
        y = output.label
        p = output.prediction
        # print(f"y:{y.shape}, p:{p.shape}")
        return self._criterion(p, y)

    def compute_batch(self, batch) -> ModelOutput:
        x,y = map(lambda v: v.to(self.device), batch)
        p = self.model(x)
        print(f"x: {x.shape}, y:{y.shape}, p:{p.shape}")
        return ModelOutput((x), y, p)  



class BatchComputerIS(IBatchComputer):
    
    def __init__(self, model, criterion, device, name = None):
        self._model = model
        self._criterion = criterion
        self.device = device
        if name is None:
            self._name =  model.__class__.__name__
        else:
            self._name = name

    @property
    def model(self):
        return self._model
    
    @property
    def name(self):
        return self._name
    
    @property
    def criterion(self):
        return self._criterion
    
    def compute_loss(self, batch) -> torch.tensor:
        output = self.compute_batch(batch)
        return self._criterion(output.prediction, output.label)

    def compute_batch(self, batch) -> ModelOutput:
        xi, yi, xr, yr = map(lambda v: v.to(self.device), batch)
        p = self.model(xi,yi,xr)
        return ModelOutput((xi, yi, xr), yr, p)  


class BatchComputerTripletLoss(IBatchComputer):
    
    def __init__(self, model, device, criterion, name = None):
        self._model = model
        self.device = device
        self._criterion = criterion 
        if name is None:
            self._name =  model.__class__.__name__
        else:
            self._name = name
        
       
    @property
    def model(self):
        return self._model

    
    @property
    def criterion(self):
        return self._criterion
    
    @property
    def name(self):
        return self._name
        
    def compute_loss(self, batch) -> torch.tensor:
        xia, yia, xip, yip, xin, yin  = map(lambda v: v.to(self.device), batch)
        
        xa = self.model(xia, yia) #anchor
        xp = self.model(xip, yip) #positive
        xn = self.model(xin, yin) #negative

        return self.criterion(xa, xp, xn)

    def compute_batch(self, batch) -> ModelOutput:
        xia, yia, xip, yip, xin, yin  = map(lambda v: v.to(self.device), batch)
        
        xa = self.model(xia, yia) #anchor
        xp = self.model(xip, yip) #positive
        xn = self.model(xin, yin) #negative

        x = torch.stack([xp,xn], 1)

        return ModelOutput((xia, yia, xip, yip, xin, yin), xa, x)  

