
#%%

from data_utils import (
    Pamap2Handler, cross_validation_split)

from default_utils import DefaultPamapPreprocessing, FcPamapPreprocessing
from preprocessing_utils import (OurConvLstmToAttentionFormat, OurConvLstmToCnnImuFormat)

from models_utils import OurConvLstmDataset, make_loader, reset_seeds
from models_utils import DatasetXY

from default_utils import TrainOurConvLSTM, TrainXY
from default_utils import make_our_conv_lstm, make_attention_transormer_model, make_fcnn
from torch import nn


base_path = ""
REPO_DIR = "."
STORE_DIR ="." 

dataset_handler = Pamap2Handler(os.path.join(REPO_DIR, ".."))

df = dataset_handler.get_protocol_subject(1)

from options import preprocessing_options

pa = preprocessing_options["AttentionTransformer"](162)

pc = preprocessing_options["OurConvLSTM"](162)

#pl = preprocessing_options["DeepConvLSTM"](162)

pf = preprocessing_options["FCNN"](162)


pa_data = pa.transformers_ts.transform(df)
pc_data = pc.transformers_ts.transform(df)
pf_data = pf.transformers_ts.transform(df)

#%%
import numpy as np

ay = np.mean(pa_data[0][:,1, 0, :], axis=1)

fy = pf_data[0][:,0, 0]


for a,b in zip(ay, fy):
    print(a,"|" ,b)
#print(pa.normdz.reverse_transform(ay), pf.normdz.reverse_transform(fy))


#%%

pf_data[0][:,0,0]

#%%

len(ay)
#%%
pa_data[0].shape, pf_data[0].shape
# %%
