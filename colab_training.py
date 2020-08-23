
#%%
import torch

args = {
    'epoch_num': 100,     # Number of epochs.
    'lr': 1.0e-3,           # Learning rate.
    'weight_decay': 10e-4, # L2 penalty.
    'momentum': 0.9,      # Momentum.
    'batch_size': 2,     # Mini-batch size. 600
    'batch_test': 2,     # size of test batch
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print(args['device'])

dataset_name = "PAMAP2"
model_type = "OurConvLSTM"
#"OurConvLSTM", "AttentionTransformer", "DeepConvLSTM", "CnnIMU"

val_sub = 0
ts_sub = 1


! pip install wget
import os
import torch
import pandas as pd
import numpy as np
import torch
from torch import nn


ssh_config = """
Host github.com
  IdentityFile ~/.ssh/github.pem
  User davipeag
  StrictHostKeyChecking no
"""

if os.name == 'nt':
  base_path = ""
  REPO_DIR = "."
  print("Windows")
else:
  print("Unix-like")
  REPO_DIR = "/tmp/HeartRateRegression"
  from google.colab import drive
  drive.mount('/content/drive')
  GIT_PATH = "/content/drive/My\ Drive/deeplearning_project/github.pem"
  DATA_PATH = "/content/drive/My\ Drive/deeplearning_project/normalized.zip"
  !mkdir ~/.ssh
  !cp -u {GIT_PATH} ~/.ssh/
  !chmod u=rw,g=,o= ~/.ssh/github.pem
  !echo "{ssh_config}" > ~/.ssh/config
  !chmod u=rw,g=,o= ~/.ssh/config
  ! (cd /tmp && git clone git@github.com:davipeag/HeartRateRegression.git)
  ! (cd {REPO_DIR} && git pull )
  import sys
  sys.path.append(REPO_DIR)


def git_push():
  if os.name == 'nt':
    pass
  else:
    ! git config --global user.email "daviaguiar@outlook.com"
    ! git config --global user.name "Davi Pedrosa de Aguiar"
    print("going to push")
    ! (cd {REPO_DIR} && git pull && cd -)
    ! (cd {REPO_DIR} && git add . && git commit -m "from colab" && git push)

def git_pull():
  if os.name == 'nt':
    pass
  else:
    ! git config --global user.email "daviaguiar@outlook.com"
    ! git config --global user.name "Davi Pedrosa de Aguiar"
    print("going to push")
    ! (cd {REPO_DIR} && git pull && cd -)
    
  
git_push()



from data_utils import (
    Pamap2Handler, cross_validation_split)

from default_utils import DefaultPamapPreprocessing
from preprocessing_utils import (OurConvLstmToAttentionFormat, OurConvLstmToCnnImuFormat)

from models_utils import OurConvLstmDataset, make_loader, reset_seeds
from models_utils import DatasetXY

from default_utils import TrainOurConvLSTM, TrainXY
from default_utils import make_our_conv_lstm, make_attention_transormer_model 
from torch import nn



reset_seeds()


#%%

dataset_handler = Pamap2Handler(os.path.join(REPO_DIR, ".."))

dfs = [dataset_handler.get_protocol_subject(s) for s in [1,2,3]]#,4,5,6,7,8]]
df_full = pd.concat(dfs)

preprocessing_options = {
    "OurConvLSTM": DefaultPamapPreprocessing(),
    "AttentionTransformer": DefaultPamapPreprocessing(last_transformer=OurConvLstmToAttentionFormat()),
    "DeepConvLSTM": DefaultPamapPreprocessing(last_transformer=OurConvLstmToCnnImuFormat()),
    "CnnIMU": DefaultPamapPreprocessing(last_transformer=OurConvLstmToCnnImuFormat())
 }

preprocessor =preprocessing_options[model_type]

#%%

preprocessor.transformers.fit(df_full)
xy_tr, xy_val, xy_ts = cross_validation_split(dfs, preprocessor.transformers, preprocessor.transformers_ts, preprocessor.transformers_ts, val_sub, ts_sub)

del dfs
del df_full

dataset_cls_options = {
    "OurConvLSTM": OurConvLstmDataset,
    "AttentionTransformer": DatasetXY,
    "DeepConvLSTM": DatasetXY,
    "CnnIMU": DatasetXY
 }


dataset_cls = dataset_cls_options[model_type]

loader_tr = make_loader(xy_tr, dataset_cls, batch_size=args["batch_size"], shuffle=True)
loader_val = make_loader(xy_val, dataset_cls, batch_size=args["batch_test"],shuffle=False)
loader_ts = make_loader(xy_ts, dataset_cls, batch_size=args["batch_test"], shuffle=False)

#%%

# ts_h_size = 32

# # ts_encoder = nn.Sequential(
# #     nn.Conv1d(40, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
# #     nn.LeakyReLU(negative_slope=0.01),
# #     nn.Dropout(),
# #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
# #     nn.LeakyReLU(negative_slope=0.01),
# #     nn.Dropout(),
# #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,)),
# #     nn.LeakyReLU(negative_slope=0.01),
# #     nn.Dropout(),
# #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
# #     nn.LeakyReLU(negative_slope=0.01),
# #     nn.Dropout(),
# #     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
# #     nn.LeakyReLU(negative_slope=0.01),
# #     nn.Dropout(),
# #     nn.Conv1d(ts_h_size, 128, kernel_size=(3,), stride=(2,)),
# #     nn.LeakyReLU(negative_slope=0.01),
# # )

# ts_h_size = 32

# ts_encoder = nn.Sequential(
#     nn.Conv1d(40, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#     nn.LeakyReLU(negative_slope=0.01),
#     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#     nn.LeakyReLU(negative_slope=0.01),
#     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,)),
#     nn.LeakyReLU(negative_slope=0.01),
#     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#     nn.LeakyReLU(negative_slope=0.01),
#     nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#     nn.LeakyReLU(negative_slope=0.01),
#     nn.Conv1d(ts_h_size, 128, kernel_size=(3,), stride=(2,)),
#     nn.Dropout(),
#     nn.LeakyReLU(negative_slope=0.01),
# )

# is_encoder = nn.Sequential(
#         nn.Conv1d(129, 32, kernel_size=(2,), stride=(2,)),
#         nn.LeakyReLU(negative_slope=0.01),
#     )

# xi,yi, xr,yr = loader_tr.__iter__().__next__()

# encoded_xp = torch.cat(
#     [net.ts_encoder(b).transpose(0,2).transpose(1,2)
#     for b in xr], dim=0)

# xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)

# encoded_xi = net.ts_encoder(xin)

# ie = torch.cat([encoded_xi, yi.transpose(2,1)], axis=1)

# i_enc = is_encoder(ie).reshape(xi.shape[0], -1)
# h = net.h0_fc_net(i_enc)
# c = net.c0_fc_net(i_enc)

# hsf, _ = net.lstm(encoded_xp, (h.unsqueeze(0),c.unsqueeze(0)))
# ps = net.predictor(net.fc_net(hsf))

# ps.shape, yr.shape



#%%
# import torch
# from torch import nn

# ts_h_size = 32

# enc = ts_encoder = nn.Sequential(
#         nn.Conv1d(40, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#         nn.LeakyReLU(negative_slope=0.01),
#         nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#         nn.LeakyReLU(negative_slope=0.01),
#         nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,)),
#         nn.LeakyReLU(negative_slope=0.01),
#         nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#         nn.LeakyReLU(negative_slope=0.01),
#         nn.Conv1d(ts_h_size, ts_h_size, kernel_size=(3,), stride=(2,), padding=(1,)),
#         nn.LeakyReLU(negative_slope=0.01),
#         nn.Conv1d(ts_h_size, 128, kernel_size=(3,), stride=(2,)),
#         nn.Dropout(),
#         nn.LeakyReLU(negative_slope=0.01),
#     )

# xin = xi.transpose(1,2).reshape(xi.shape[0], xi.shape[2],  -1)
# enc(xin).shape, xin.shape
#%%
from default_utils import make_cnn_imu2
from default_utils import make_deep_conv_lstm


net_options = {
    "OurConvLSTM": lambda : make_our_conv_lstm(40,1,False),
    "AttentionTransformer": lambda: make_attention_transormer_model(args["device"]),
    "DeepConvLSTM": lambda : make_deep_conv_lstm(),
    "CnnIMU": lambda : make_cnn_imu2()
}

net = net_options[model_type]().to(args["device"])
criterion = nn.L1Loss().to(args["device"]) 
optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"],
                             weight_decay=args["weight_decay"])

#%%


basic_training_parameters = {
    "net": net,
    "criterion": criterion,
    "optimizer": optimizer,
    "loader_tr": loader_tr,
    "loader_val": loader_val,
    "loader_ts": loader_ts,
    "normdz": preprocessor.normdz,
    "ztransformer": preprocessor.ztransformer,
    "device": args["device"]
}


trainer_options = {
    "OurConvLSTM": lambda : TrainOurConvLSTM(**basic_training_parameters),
    "AttentionTransformer": lambda : TrainXY(
        **basic_training_parameters,
        get_last_y_from_x = lambda x: x[:,1, 0, -1].reshape(-1,1)
    ),
    "CnnIMU":lambda : TrainXY(
        **basic_training_parameters,
        get_last_y_from_x = lambda x: np.mean(x[:,0,200:300, 0], axis=1).reshape(-1,1)
    ),
    "DeepConvLSTM":lambda : TrainXY(
        **basic_training_parameters,
        get_last_y_from_x = lambda x: np.mean(x[:,0,200:300, 0], axis=1).reshape(-1,1)
    ), 

}

trainer = trainer_options[model_type]()


# %%

run_output = trainer.train_epochs(args["epoch_num"])


#%%

state_dict_name = f"trained_models/{model_type}ts_{ts_sub}_val_{val_sub}.pkl"
torch.save(run_output["best_val_model"], os.path.join(REPO_DIR, state_dict_name))

git_push()

# %%
