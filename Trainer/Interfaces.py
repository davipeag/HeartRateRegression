import torch
import numpy as np
from abc import abstractmethod, ABC
import copy

from typing import Sequence, Tuple, List, Dict, Callable


class ModelOutput:
    def __init__(self, features: Tuple[torch.tensor], label: torch.tensor, prediction: torch.tensor):
        self.features = features
        self.label = label
        self.prediction = prediction
    
    def to_numpy(self):
        if isinstance(self.prediction, np.ndarray):
            return self
        self.features = [f.detach().cpu().numpy() for f in self.features]
        if self.label is not None: self.label = self.label.detach().cpu().numpy()
        self.prediction = self.prediction.detach().cpu().numpy()
        return self


class IBatchComputer(ABC):

    @property
    @abstractmethod
    def model(self):
        pass
    
    @property
    @abstractmethod
    def name(self):
        pass
    
    @abstractmethod
    def compute_loss(self, batch) -> torch.float:
        pass

    @abstractmethod
    def compute_batch(self, batch) -> ModelOutput:
        pass


class IBatchMultiTrainer(ABC):
    @property
    @abstractmethod
    def names(self) -> Tuple[str]:
        pass

    @property
    @abstractmethod
    def models(self) -> Tuple:
        pass

    @abstractmethod
    def train_batch(self, batches: Sequence) -> float:
        pass

    @abstractmethod
    def evaluate_batch(self, batches: Sequence) -> Tuple[float]:
        pass
    
    @abstractmethod
    def compute_batch(self, batches: Sequence) -> Tuple[ModelOutput]:
        pass


    
