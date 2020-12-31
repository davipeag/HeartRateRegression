from Trainer.Interfaces import IBatchComputer, IBatchMultiTrainer, ModelOutput
from typing import Sequence, Tuple, List
import torch


class SequentialTrainer(IBatchMultiTrainer):

    def __init__(self, computers: Sequence[IBatchMultiTrainer], optimizer, weights: Sequence[float] = None):
        self._computers = computers
        self._optimizer = optimizer
        if weights is None:
            self._weights = [1 for _ in computers]
        else:
            assert len(weights) == len(computers)
            self._weights = weights

    @property
    def names(self) -> Tuple[str]:
        return [computer.name for computer in self._computers]

    @property
    def models(self) -> Tuple:
        return [computer.model for computer in self._computers]

    def train_batch(self, batches: Sequence) -> Tuple[float]:
        [model.train() for model in self.models]
        [model.zero_grad() for model in self.models]
        losses  = [w*computer.compute_loss(b) for w, computer,
                          b in zip(self._weights, self._computers, batches)]
        loss = losses[0]
        for partial_loss in losses[1:]: loss = loss + partial_loss

        loss.backward()
        self._optimizer.step()
        return loss.cpu().item()

    def evaluate_batch(self, batches: Sequence) -> Tuple[float]:
        [model.eval() for model in self.models]
        with torch.no_grad():
            loss = [computer.compute_loss(b).cpu().item(
            ) for computer, b in zip(self._computers, batches)]
        return loss

    def compute_batch(self, batches: Sequence) -> Tuple[ModelOutput]:
        [model.eval() for model in self.models]
        with torch.no_grad():
            outputs = [computer.compute_batch(b).to_numpy(
            ) for computer, b in zip(self._computers, batches)]
        return outputs
    
    def single_compute_batch(self, batch, idx) -> ModelOutput:
        return self._computers[idx].compute_batch(batch)
    
    def single_evaluate_batch(self, batch, idx) -> float:
        self.models[idx].eval()
        with torch.no_grad():
            return self._computers[idx].compute_loss(batch)
    
    def single_train_batch(self, batch, idx) -> float:
        model = self.models[idx]
        computer = self._computers[idx]
        weight = self._weights[idx]
        model.train()
        model.zero_grad()
        loss  = weight*computer.compute_loss(batch)
        loss.backward()
        self._optimizer.step()
        return loss.cpu().item()

