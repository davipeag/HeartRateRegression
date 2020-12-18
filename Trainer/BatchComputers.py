from Trainer.Interfaces import IBatchComputer, ModelOutput
import torch

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

