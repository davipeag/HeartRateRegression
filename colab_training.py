
#%%
import torch

args = {
    'epoch_num': 100,     # Number of epochs.
    'lr': 1.0e-3,           # Learning rate.
    'weight_decay': 10e-4, # L2 penalty.
    'momentum': 0.9,      # Momentum.
    'batch_size': 1024,     # Mini-batch size. 600
    'batch_test': 1024,     # size of test batch
}

if torch.cuda.is_available():
    args['device'] = torch.device('cuda')
else:
    args['device'] = torch.device('cpu')

print(args['device'])

dataset_name = "PAMAP2"
model_type = "FCNN"
#"OurConvLSTM", "AttentionTransformer", "DeepConvLSTM", "CnnIMU", FCNN

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
  STORE_DIR ="." 
  print("Windows")
else:
  print("Unix-like")
  REPO_DIR = "/tmp/HeartRateRegression"
  from google.colab import drive
  drive.mount('/content/drive')
  GIT_PATH = "/content/drive/My\ Drive/deeplearning_project/github.pem"
  DATA_PATH = "/content/drive/My\ Drive/deeplearning_project/normalized.zip"
  STORE_DIR ="/content/drive/My\ Drive/deeplearning_project/" 
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

from default_utils import DefaultPamapPreprocessing, FcPamapPreprocessing
from preprocessing_utils import (OurConvLstmToAttentionFormat, OurConvLstmToCnnImuFormat)

from models_utils import OurConvLstmDataset, make_loader, reset_seeds
from models_utils import DatasetXY

from default_utils import TrainOurConvLSTM, TrainXY
from default_utils import make_our_conv_lstm, make_attention_transormer_model, make_fcnn
from torch import nn



reset_seeds()

##%%

dataset_handler = Pamap2Handler(os.path.join(REPO_DIR, ".."))

dfs = [dataset_handler.get_protocol_subject(s) for s in [1,2,3,]]#4,5,6,7,8]]
df_full = pd.concat(dfs)

preprocessing_options = {
    "OurConvLSTM": DefaultPamapPreprocessing(ts_count = 300, donwsampling_ratio = 1),
    "AttentionTransformer": DefaultPamapPreprocessing(
      ts_count = 300, donwsampling_ratio = 1, last_transformer=OurConvLstmToAttentionFormat()),
    "DeepConvLSTM": DefaultPamapPreprocessing(last_transformer=OurConvLstmToCnnImuFormat()),
    "CnnIMU": DefaultPamapPreprocessing(last_transformer=OurConvLstmToCnnImuFormat()),
    "FCNN": FcPamapPreprocessing(),
 }

preprocessor =preprocessing_options[model_type]

##%%

preprocessor.transformers.fit(df_full)
xy_tr, xy_val, xy_ts = cross_validation_split(dfs, preprocessor.transformers, preprocessor.transformers_ts, preprocessor.transformers_ts, val_sub, ts_sub)

del dfs
del df_full

dataset_cls_options = {
    "OurConvLSTM": OurConvLstmDataset,
    "AttentionTransformer": DatasetXY,
    "DeepConvLSTM": DatasetXY,
    "CnnIMU": DatasetXY,
    "FCNN": DatasetXY,
 }


dataset_cls = dataset_cls_options[model_type]

loader_tr = make_loader(xy_tr, dataset_cls, batch_size=args["batch_size"], shuffle=True)
loader_val = make_loader(xy_val, dataset_cls, batch_size=args["batch_test"],shuffle=False)
loader_ts = make_loader(xy_ts, dataset_cls, batch_size=args["batch_test"], shuffle=False)

# #%%
# x,y = loader_tr.__iter__().__next__()
# x.shape, y.shape

##%%
from default_utils import make_cnn_imu2
from default_utils import make_deep_conv_lstm


net_options = {
    "OurConvLSTM": lambda : make_our_conv_lstm(40,1,False),
    "AttentionTransformer": lambda: make_attention_transormer_model(args["device"]),
    "DeepConvLSTM": lambda : make_deep_conv_lstm(),
    "CnnIMU": lambda : make_cnn_imu2(),
    "FCNN": lambda : make_fcnn()
}

net = net_options[model_type]().to(args["device"])
criterion = nn.L1Loss().to(args["device"]) 
optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"],
                             weight_decay=args["weight_decay"])

##%%


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
    "FCNN":lambda : TrainXY(
        **basic_training_parameters,
        get_last_y_from_x = lambda x: x[:,0, 0].reshape(-1,1)
    ), 

}

trainer = trainer_options[model_type]()


# %%
import matplotlib.pyplot as plt

x,y = loader_ts.__iter__().__next__()
x.shape
xc = torch.cat([x[:, j, :, :] for j in range(x.shape[1])],dim=2)

for i in [2,3,4,5,6,7]:
  #i = 7 
  count = 30
  n = 300*count + 600
  fig, ax = plt.subplots(figsize=(60,5))
  s = xc[0,i,600:n]
  ax.plot(s, '-k')
  #ax.plot(y[0,:count,0], 'ok')
  ax.axis("off")
  fig.savefig(f"figures/s{i}.jpg")
#fig.savefig(f"figures/heart_rate.jpg")
#%%
y.shape
#plot_s(0)

#%%
run_output = trainer.train_epochs(args["epoch_num"])


#%%

state_dict_name = f"trained_models/{model_type}ts_{ts_sub}_val_{val_sub}.pkl"
torch.save(run_output["best_val_model"], os.path.join(STORE_DIR, state_dict_name))

git_push()

# %%
