
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
model_type = "AttentionTransformer"
#"OurConvLSTM", "AttentionTransformer", "DeepConvLSTM", "CnnIMU"

plot_subject = 2

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

#%%
preprocessing_options = {
    "OurConvLSTM": DefaultPamapPreprocessing(),
    "AttentionTransformer": DefaultPamapPreprocessing(last_transformer=OurConvLstmToAttentionFormat()),
    "DeepConvLSTM": DefaultPamapPreprocessing(last_transformer=OurConvLstmToCnnImuFormat()),
    "CnnIMU": DefaultPamapPreprocessing(last_transformer=OurConvLstmToCnnImuFormat())
 }

models = ["AttentionTransformer", "OurConvLSTM"]

preprocessors =[preprocessing_options[model] for model in models]
tdata = list()
for preprocessor in preprocessors:
    preprocessor.transformers.fit(df_full)
    tdata.append(preprocessor.transformers.transform(dfs[plot_subject]))

del dfs
del df_full

dataset_cls_options = {
    "OurConvLSTM": OurConvLstmDataset,
    "AttentionTransformer": DatasetXY,
    "DeepConvLSTM": DatasetXY,
    "CnnIMU": DatasetXY
 }


datasets_cls = [dataset_cls_options[model_type] for model_type in models]

loaders = [make_loader([xy], dataset_cls, batch_size = args["batch_size"], shuffle=False)
            for dataset_cls, xy in zip(datasets_cls, tdata)]


#%%
from default_utils import make_cnn_imu2
from default_utils import make_deep_conv_lstm


net_options = {
    "OurConvLSTM": lambda : make_our_conv_lstm(40,1,False),
    "AttentionTransformer": lambda: make_attention_transormer_model(args["device"]),
    "DeepConvLSTM": lambda : make_deep_conv_lstm(),
    "CnnIMU": lambda : make_cnn_imu2()
}

nets = [net_options[model_type]().to(args["device"]) for model_type in models]

#net = net_options[model_type]().to(args["device"])
criterion = nn.L1Loss().to(args["device"]) 


#%%


trainers = list()

for preprocessor,net, model_type in zip(preprocessors,nets, models):

    optimizer = torch.optim.Adam(net.parameters(), lr=args["lr"],
                             weight_decay=args["weight_decay"])

    basic_training_parameters = {
        "net": net,
        "criterion": criterion,
        "optimizer": optimizer,
        "loader_tr": None,
        "loader_val": None,
        "loader_ts": None,
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
    trainers.append(trainer)

import os
import matplotlib.pyplot as plt
plt.figure()
for model_type, loader, trainer in zip(models, loaders, trainers):
    state_dict_name = f"trained_models/{model_type}ts_{plot_subject}_val_{4}.pkl"
    trainer.net.load_state_dict(torch.load(state_dict_name, map_location=args["device"]))
    y,p = reverse_transformed_prediction_labels()
    plt.plot(p, label=model_type)
plt.plot(y, label="actual")
plt.labels()
plt.show()

# %%

run_output = trainer.train_epochs(args["epoch_num"])


#%%

state_dict_name = f"trained_models/{model_type}ts_{ts_sub}_val_{val_sub}.pkl"
torch.save(run_output["best_val_model"], os.path.join(REPO_DIR, state_dict_name))

git_push()

# %%
