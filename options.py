

#%%
import numpy as np
from default_utils import (
    DefaultPamapPreprocessing,
    FcPamapPreprocessing,
    make_deep_conv_lstm,
    make_cnn_imu2,
    make_fcnn,
    make_our_conv_lstm,
    make_attention_transormer_model,
    TrainOurConvLSTM,
    TrainXY
)

from preprocessing_utils import (
    ShuffleIS,
    OurConvLstmToAttentionFormat,
    OurConvLstmToCnnImuFormat
)

from models_utils import (
    OurConvLstmDataset,
    DatasetXY
)


preprocessing_options = {
      "OurConvLSTM": lambda ts_per_sample = 162:  DefaultPamapPreprocessing(
          ts_per_sample=ts_per_sample, ts_count = 300, donwsampling_ratio = 1),
      "AttentionTransformer": lambda ts_per_sample = 162: DefaultPamapPreprocessing(
        ts_per_sample=ts_per_sample, ts_count = 300, donwsampling_ratio = 1, last_transformer=OurConvLstmToAttentionFormat()),
      "DeepConvLSTM": lambda ts_per_sample = 162: DefaultPamapPreprocessing(
          ts_per_sample=ts_per_sample, last_transformer=OurConvLstmToCnnImuFormat()),
      "CnnIMU": lambda ts_per_sample = 162: DefaultPamapPreprocessing(
          ts_per_sample=ts_per_sample, last_transformer=OurConvLstmToCnnImuFormat()),
      "FCNN": lambda ts_per_sample = 162: FcPamapPreprocessing(ts_per_sample=ts_per_sample,),
      "NoISOurConvLSTM": lambda ts_per_sample = 162: DefaultPamapPreprocessing(ts_per_sample=ts_per_sample,ts_count = 300, donwsampling_ratio = 1),
      "ShuffleISOurConvLSTM": lambda ts_per_sample = 162: DefaultPamapPreprocessing(ts_per_sample=ts_per_sample,ts_count = 300, donwsampling_ratio = 1, last_transformer=ShuffleIS),
      "LstmISOurConvLSTM": lambda ts_per_sample = 162: DefaultPamapPreprocessing(ts_per_sample=ts_per_sample, ts_count = 300, donwsampling_ratio = 1)
}

dataset_cls_options = {
      "OurConvLSTM": OurConvLstmDataset,
      "AttentionTransformer": DatasetXY,
      "DeepConvLSTM": DatasetXY,
      "CnnIMU": DatasetXY,
      "FCNN": DatasetXY,
      "NoISOurConvLSTM": OurConvLstmDataset,
      "ShuffleISOurConvLSTM": OurConvLstmDataset,
      "LstmISOurConvLSTM": OurConvLstmDataset,
  }


net_options = {
      "OurConvLSTM": lambda t=162,r=160 : make_our_conv_lstm(40,1),
      "AttentionTransformer": lambda t=162,r=160: make_attention_transormer_model(args["device"]),
      "DeepConvLSTM": lambda t=162,r=160 : make_deep_conv_lstm(total_size=t, recursive_size=4),
      "CnnIMU": lambda t=162,r=160 : make_cnn_imu2(total_size=t, recursive_size=4),
      "FCNN": lambda t=162,r=160 : make_fcnn(),
      "NoISOurConvLSTM":lambda t=162,r=160: make_our_conv_lstm(40,1,"nois"),
      "ShuffleISOurConvLSTM": lambda t=162,r=160 : make_our_conv_lstm(40,1,"regular"),
      "LstmISOurConvLSTM": lambda t=162,r=160: make_our_conv_lstm(40,1, "lstmis"),
  }



trainer_options = {
      "OurConvLSTM": lambda pars : TrainOurConvLSTM(**pars),
      "AttentionTransformer": lambda pars: TrainXY(
          **pars, get_last_y_from_x = lambda x: x[:,1, 0, -1].reshape(-1,1)
      ),
      "CnnIMU":lambda pars: TrainXY(
          **pars,
          get_last_y_from_x = lambda x: np.mean(x[:,0,200:300, 0], axis=1).reshape(-1,1)
      ),
      "DeepConvLSTM":lambda pars : TrainXY(
          **pars,
          get_last_y_from_x = lambda x: np.mean(x[:,0,200:300, 0], axis=1).reshape(-1,1)
      ),
      "FCNN":lambda pars: TrainXY(
          **pars,
          get_last_y_from_x = lambda x: x[:,0, 0].reshape(-1,1)
      ),
      "NoISOurConvLSTM":lambda pars: TrainOurConvLSTM(**pars),

      "ShuffleISOurConvLSTM": lambda pars: TrainOurConvLSTM(**pars),

      "LstmISOurConvLSTM": lambda pars: TrainOurConvLSTM(**pars),
  }

