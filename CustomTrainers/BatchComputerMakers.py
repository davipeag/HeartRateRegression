import torch
from Trainer.BatchComputers import BatchComputerIS
import PPG

class PceLstmBatchComputerMaker:
    def __init__(
        self, lstm_builder_cls, device, input_features_parameter_name, feature_count,
        additional_net_args = dict(),
        criterion = torch.nn.L1Loss(),
        name = "PceLstm"
        ):
        self.device = device
        self.lstm_builder_cls = lstm_builder_cls
        self.input_features_parameter_name = input_features_parameter_name
        self.additional_net_args = additional_net_args
        self.criterion = criterion
        self.feature_count = feature_count
        self.name = name
    
    def make(self, sample_per_ts, ts_per_is, **net_args):
        net_args[self.input_features_parameter_name] = self.feature_count
        net_args["sample_per_ts"] = sample_per_ts
        net_args["ts_per_is"] = ts_per_is
        net_args = {**net_args, **self.additional_net_args}
        
        
        lstm = self.lstm_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(lstm)        
        
        return BatchComputerIS(lstm, self.criterion, self.device, self.name)
        
class NoPceLstmBatchComputerMaker:
    def __init__(
        self, lstm_builder_cls, device, input_features_parameter_name, feature_count,
        additional_net_args = dict(),
        criterion = torch.nn.L1Loss(),
        name = "PceLstm"
        ):
        self.device = device
        self.lstm_builder_cls = lstm_builder_cls
        self.input_features_parameter_name = input_features_parameter_name
        self.additional_net_args = additional_net_args
        self.criterion = criterion
        self.feature_count = feature_count
        self.name = name
    
    def make(self, sample_per_ts, **net_args):
        net_args[self.input_features_parameter_name] = self.feature_count
        net_args["sample_per_ts"] = sample_per_ts
        net_args = {**net_args, **self.additional_net_args}
        
        
        lstm = self.lstm_builder_cls(**net_args).to(self.device)
        PPG.Models.initialize_weights(lstm)        
        
        return BatchComputerIS(lstm, self.criterion, self.device, self.name)

class DeepPceDiscriminatorBatchComputerMaker:
    def __init__(self, discriminator_builder_cls, device,
                        criterion = torch.nn.BCEWithLogitsLoss(),
                        name="Discriminator"
                ):
        self.discriminator_builder_cls = discriminator_builder_cls
        self.criterion = criterion
        self.name = name
        self.device = device
      
    def make(self, ts_encoder, is_encoder, **net_args):
        net_args["ts_encoder"] = ts_encoder
        net_args["is_encoder"] = is_encoder

        discriminator = self.discriminator_builder_cls(**net_args)
        PPG.Models.initialize_weights(discriminator)
        return BatchComputerIS()
