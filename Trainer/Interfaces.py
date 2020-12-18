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
        self.prediction = self.prediction.detach().cpu().numpy
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
    def train_batch(self, batches: Sequence) -> Tuple[float]:
        pass

    @abstractmethod
    def evaluate_batch(self, batches: Sequence) -> Tuple[float]:
        pass
    
    @abstractmethod
    def compute_batch(self, batches: batches) -> Tuple[ModelOutput]:
        pass

class EpochTrainer():    
    
    def __init__(self, batch_trainer: IBatchMultiTrainer):
        self.batch_trainer = batch_trainer

    def required_loader_count(self):
        return len(self.batch_trainer.names)
    
    @property
    def names(self):
        return self.batch_trainer.names
    
    @property
    def models(self):
        return self.batch_trainer.names

    def apply_to_batches(self, function, loaders):
        return [function(batches) for batches in zip(*loaders)]

    def train_epoch(self, loaders: Sequence) -> List[Tuple[str, np.ndarray]]:
        losses = zip(*self.apply_to_batches(self.batch_trainer.train_batch, loaders))
        return zip(self.batch_trainer.names, losses)


    def evaluate_epoch(self, loaders: Sequence) -> List[Tuple[str, np.ndarray]]:
        losses = zip(*self.apply_to_batches(self.batch_trainer.evaluate_batch, loaders))
        return zip(self.batch_trainer.names, losses)
    
    def compute_epoch(self, loaders) -> List[ModelOutput]:
        outputs = zip(*self.apply_to_batches(self.batch_trainer.compute_batch, loaders))
        labels = [np.concatenate([out.to_numpy().label for out in moutputs]) for moutputs in outputs]
        predictions = [np.concatenate([out.prediction for out in moutputs]) for moutputs in outputs]
        features = [[np.concatenate(f) for f in zip(*[out.features for out in moutputs])] for moutputs in outputs]
        return [ModelOutput(f, l, p) for f,l,p in zip(features, labels, predictions)]

      

class TrainHelperJoint():
    def __init__(
            self, trainer: EpochTrainer,
            loaders_train: List,
            loaders_validation: List,
            loaders_test,
            display_criterions: Callable[[ModelOutput], float],
            optimizing_model_index = 0
        ):
        
        self.trainer = trainer
        self.loaders_tr = loaders_train
        self.loaders_ts = loaders_test
        self.loaders_val = loaders_validation
        self.display_criterions = display_criterions
        self.optimizing_model_index = optimizing_model_index


 
    def compute_metric(self, loaders):
        return [display_criterion(output) for output, display_criterion in zip(
            self.trainer.evaluate_epoch(loaders), self.display_criterions)]
    
    def compute_outputs(self, loaders):
        return self.trainer.evaluate_epoch(loaders)
    
    def get_models(self):
        return self.trainer.models
    
    def get_state_dicts(self):
        return [copy.deepcopy(m.state_dict()) for m in self.get_models()]

    def load_state_dicts(self, state_dicts):
        return [m.load_state_dict(sd) for m, sd in zip(self.get_models(), state_dicts)]


    def train(self, n_epoch):
        best_val_models  = self.get_state_dicts()
        
        validation_metrics = [self.compute_metric(self.loaders_val)] 
        test_metric = self.compute_metric(self.loaders_ts) 
    
        for epoch in range(1, n_epoch+1):
            self.trainer.train(*self.loaders_tr)
            loss_val = self.compute_metric(self.loaders_val)
            
            if loss_val[self.optimizing_model_index] < np.min([v[0] for v in validation_metrics]):
                print("best val epoch:", epoch)
                loss_tr = self.compute_metric(self.loaders_tr)
                loss_ts = self.compute_metric(self.loaders_ts)
                outputs_ts = self.compute_outputs(loaders_ts)
                test_metric = loss_ts 
                best_val_models = self.get_state_dicts()# copy.deepcopy(self.trainer.model.state_dict())
                print(f'[{epoch}/{n_epoch}]: loss_train: {loss_tr} loss_val {loss_val} loss_ts {loss_ts}' )
            validation_metrics.append(loss_val)

      
        self.load_state_dicts(best_val_models)
        print(f"Final: {"labels": test_metric.label, "predictions": test_metric.prediction}")
        return  {"labels": test_metric.label, "predictions": test_metric.prediction }

    
    
