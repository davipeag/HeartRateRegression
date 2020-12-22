
from Trainer.Interfaces import *


class MultiModelEpochTrainer():    
    
    def __init__(self, batch_trainer: IBatchMultiTrainer):
        self.batch_trainer = batch_trainer

    def required_loader_count(self):
        return len(self.batch_trainer.names)
    
    @property
    def names(self):
        return self.batch_trainer.names
    
    @property
    def models(self):
        return self.batch_trainer.models

    def apply_to_batches(self, function, loaders):
        return [function(batches) for batches in zip(*loaders)]

    def train_epoch(self, loaders: Sequence) -> List[np.ndarray]:
        losses = self.apply_to_batches(self.batch_trainer.train_batch, loaders)
        return losses #zip(self.batch_trainer.names, losses)


    def evaluate_epoch(self, loaders: Sequence) -> List[np.ndarray]:
        losses = zip(*self.apply_to_batches(self.batch_trainer.evaluate_batch, loaders))
        return list(losses)
    
    def compute_epoch(self, loaders) -> List[ModelOutput]:
        outputs = list(zip(*self.apply_to_batches(self.batch_trainer.compute_batch, loaders)))
        labels = [np.concatenate([out.to_numpy().label for out in moutputs]) for moutputs in outputs]
        predictions = [np.concatenate([out.prediction for out in moutputs]) for moutputs in outputs]
        features = [[np.concatenate(f) for f in zip(*[out.features for out in moutputs])] for moutputs in outputs]
        return [ModelOutput(f, l, p) for f,l,p in zip(features, labels, predictions)]

      
class MultiModelTrainHelper():
    def __init__(
            self, trainer: MultiModelEpochTrainer,
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
            self.trainer.compute_epoch(loaders), self.display_criterions)]
    
    def compute_outputs(self, loaders):
        return self.trainer.compute_epoch(loaders)
    
    def get_models(self):
        return self.trainer.models
    
    def get_state_dicts(self):
        return [copy.deepcopy(m.state_dict()) for m in self.get_models()]

    def load_state_dicts(self, state_dicts):
        return [m.load_state_dict(sd) for m, sd in zip(self.get_models(), state_dicts)]


    def train(self, n_epoch):
        best_val_models  = self.get_state_dicts()
        
        validation_metrics = [self.trainer.evaluate_epoch(self.loaders_val)] #self.compute_metric(self.loaders_val)]  
    
        for epoch in range(1, n_epoch+1):
            self.trainer.train_epoch(self.loaders_tr)
            loss_val = self.trainer.evaluate_epoch(self.loaders_val)# self.compute_metric(self.loaders_val)
            
            if loss_val[self.optimizing_model_index] < np.min([v[self.optimizing_model_index] for v in validation_metrics]):
                print("best val epoch:", epoch)
                loss_tr = self.compute_metric(self.loaders_tr)
                loss_val_ptr = self.compute_metric(self.loaders_val)
                outputs_ts = self.compute_outputs(self.loaders_ts)
                loss_ts = [display_criterion(output) for output, display_criterion in zip(
                        outputs_ts, self.display_criterions)]
                best_val_models = self.get_state_dicts()
                print(f'[{epoch}/{n_epoch}]: loss_train: {loss_tr} loss_val {loss_val_ptr} loss_ts {loss_ts}' )
            validation_metrics.append(loss_val)
        
        final_output = dict()
        for model_output, metric, name in zip(outputs_ts, loss_ts, self.trainer.names):
            out = {"labels": model_output.label, "predictions": model_output.prediction, "metric": metric }
            final_output[name] = out       
      
        self.load_state_dicts(best_val_models)
        print(f"Final: {loss_ts}")
        return  final_output

    

