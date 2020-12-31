from . import FullTrainers

import Models
from Models import TimeSnippetModels
from PreprocessingHelpers import TransformerGetters
from Constants  import DatasetMapping

import PPG

from PPG import NoHrPceLstmModel
from RegressionHR import PceLstmModel

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


class SingleNoHrPpgPceLstmFullTrainer(FullTrainers.SingleNetFullTrainerJointValidationIS):
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
        args_to_net_args_mapping = {"ts_per_is": "ts_per_is", "sample_per_ts": "sample_per_ts"},
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


class SinglePceLstmFullTrainerHandChestAccelerometers(FullTrainers.SingleNetFullTrainerJointValidationIS):
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        dataset_name,
        model_name = None,
        frequency_hz = None,
        net_builder_cls = PceLstmModel.make_par_enc_pce_lstm,
        transformer_getter_cls = TransformerGetters.PceLstmTransformerGetterRenamed,
        input_features_parameter_name = "nattrs",
        additional_args = dict(),
        args_to_net_args_mapping = {"ts_per_is": "ts_per_is", "sample_per_ts": "sample_per_ts"},
        args_function_mapping = {"sample_per_ts": lambda a: a["frequency_hz"]*a["period_s"]}
        ):
        if dataset_name == "pamap2":
            feature_columns = [
            'heart_rate', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6']
        else:
            feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        # feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        if frequency_hz == None:
            frequency_hz = DatasetMapping.FrequencyMapping[dataset_name]

        print(f"frequency_hz: {frequency_hz}")

      
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

class SingleNoPceLstmFullTrainerHandChestAccelerometers(FullTrainers.SingleNetFullTrainerJointValidationIS):
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        dataset_name,
        model_name = None,
        frequency_hz = None,
        net_builder_cls = PceLstmModel.make_par_enc_no_pce_lstm,
        transformer_getter_cls = TransformerGetters.PceLstmTransformerGetterRenamed,
        input_features_parameter_name = "nattrs",
        additional_args = dict(),
        args_to_net_args_mapping = {"ts_per_is": "ts_per_is", "sample_per_ts": "sample_per_ts"},
        args_function_mapping = {"sample_per_ts": lambda a: a["frequency_hz"]*a["period_s"]}
        ):
        if dataset_name == "pamap2":
            feature_columns = [
            'heart_rate', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6']
        else:
            feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        # feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        if frequency_hz == None:
            frequency_hz = DatasetMapping.FrequencyMapping[dataset_name]

        print(f"frequency_hz: {frequency_hz}")

      
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

class SinglePceLstmFullTrainerIMU(FullTrainers.SingleNetFullTrainerJointValidationIS):
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        dataset_name,
        model_name = None,
        frequency_hz = None,
        net_builder_cls = PceLstmModel.make_par_enc_pce_lstm,
        transformer_getter_cls = TransformerGetters.PceLstmTransformerGetterRenamed,
        input_features_parameter_name = "nattrs",
        additional_args = dict(),
        args_to_net_args_mapping = {"ts_per_is": "ts_per_is"},
        args_function_mapping = {"sample_per_ts": lambda a: a["frequency_hz"]*a["period_s"]}
        ):
        if dataset_name == "pamap2":
            feature_columns = [
            'heart_rate', 'h_xacc16', 'h_yacc16', 'h_zacc16',
            'h_xacc6', 'h_yacc6', 'h_zacc6', 'c_xacc16', 'c_yacc16', 'c_zacc16',
            'c_xacc6', 'c_yacc6', 'c_zacc6']
        else:
            feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        # feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        if frequency_hz == None:
            frequency_hz = DatasetMapping.FrequencyMapping[dataset_name]

        print(f"frequency_hz: {frequency_hz}")

      
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


class SinglePceLstmFullTrainerMotionIMU(FullTrainers.SingleNetFullTrainerJointValidationIS):
    def __init__(
        self,
        dfs,
        device,
        nepoch,
        dataset_name,
        model_name = None,
        frequency_hz = None,
        net_builder_cls = PceLstmModel.make_par_enc_pce_lstm,
        transformer_getter_cls = TransformerGetters.PceLstmTransformerGetterRenamed,
        input_features_parameter_name = "nattrs",
        additional_args = dict(),
        args_to_net_args_mapping = {"ts_per_is": "ts_per_is"},
        args_function_mapping = {"sample_per_ts": lambda a: a["frequency_hz"]*a["period_s"]}
        ):
        if dataset_name == "pamap2":
            feature_columns = [
                'heart_rate', 'h_xacc16', 'h_yacc16', 'h_zacc16',
                'h_xacc6', 'h_yacc6', 'h_zacc6', 'h_xgyr', 'h_ygyr', 'h_zgyr',
                'c_xacc16', 'c_yacc16', 'c_zacc16', 'c_xacc6', 'c_yacc6', 'c_zacc6',
                'c_xgyr', 'c_ygyr', 'c_zgyr',
                'a_xacc16', 'a_yacc16', 'a_zacc16',
                'a_xacc6', 'a_yacc6', 'a_zacc6', 'a_xgyr', 'a_ygyr', 'a_zgyr'
            ]
        else:
            feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        # feature_columns = DatasetMapping.NoPpgFeatureColumns[dataset_name]
        if frequency_hz == None:
            frequency_hz = DatasetMapping.FrequencyMapping[dataset_name]

        print(f"frequency_hz: {frequency_hz}")

      
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