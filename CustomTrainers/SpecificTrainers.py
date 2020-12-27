from . import FullTrainers

import Models
from Models import TimeSnippetModels
from PreprocessingHelpers import TransformerGetters
from Constants  import DatasetMapping

import PPG

from PPG import NoHrPceLstmModel

class DeepConvLstmFullTrainer(FullTrainers.SingleNetFullTrainerJointValidationXY):
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        dataset_name,
        model_name = None,
        net_builder_cls = TimeSnippetModels.DeepConvLSTM,
        transformer_getter_cls = TransformerGetters.DeepConvLstmTransformerGetter ,
        frequency_hz = 100/3,
        input_features_parameter_name = "feature_count",
        additional_args = dict(),
        args_to_net_args_mapping = {"ts_per_is": "ts_per_is", "period_seconds": "period_s", "ts_per_window": "ts_per_window",
                                    "frequency_hz": "frequency_hz"},
        args_function_mapping = dict()
        ):
        feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]

      
        super().__init__(
            dfs = dfs,
            device = device,
            nepoch = nepoch,
            net_builder_cls = net_builder_cls,
            transformer_getter_cls = transformer_getter_cls,
            dataset_name = dataset_name,
            feature_columns = feature_columns,
            frequency_hz = frequency_hz,
            input_features_parameter_name = input_features_parameter_name,
            additional_args = additional_args,
            args_to_net_args_mapping = args_to_net_args_mapping,
            args_function_mapping = args_function_mapping,
            model_name = model_name
        )


class SingleNoHrPpgPceLstmFullTrainer(FullTrainers.SingleNetFullTrainerJointValidationXY):
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        dataset_name,
        model_name = None,
        net_builder_cls = NoHrPceLstmModel.ppg_make_par_enc_pce_lstm,
        transformer_getter_cls = TransformerGetters.PpgPceLstmTransformerGetter ,
        input_features_parameter_name = "nattrs",
        additional_args = dict(),
        args_to_net_args_mapping = {"ts_per_is": "ts_per_is", "period_s": "period_s", "ts_per_window": "ts_per_window",
                                    "sample_per_ts": "sample_per_ts"},
        args_function_mapping = {"sample_per_ts": lambda a: a["frequency_hz"]*a["period_s"]}
        ):
        feature_columns = DatasetMapping.PpgFeatureColumns[dataset_name]
        frequency_hz = DatasetMapping.FrequencyMapping[dataset_name]

      
        super().__init__(
            dfs = dfs,
            device = device,
            nepoch = nepoch,
            net_builder_cls = net_builder_cls,
            transformer_getter_cls = transformer_getter_cls,
            dataset_name = dataset_name,
            feature_columns = feature_columns,
            frequency_hz = frequency_hz,
            input_features_parameter_name = input_features_parameter_name,
            additional_args = additional_args,
            args_to_net_args_mapping = args_to_net_args_mapping,
            args_function_mapping = args_function_mapping,
            model_name = model_name
        )