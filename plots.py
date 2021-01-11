#%%
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

path = "executions/pamap2results.pkl"

with open(path, "rb") as f:
    results = pickle.load(f)

#%%
results.keys()

#%%
nodisc = results['PCE-LSTM discriminator']
d_metrics = dict()
for idx in nodisc.keys():
    ind = nodisc[idx]    
    d_metrics[idx + 1] = [v["_Discriminator"]["metric"].item() for v in ind.values()]
    
d_metrics

#%%


def get_mae(p,y): return np.mean(np.abs(p-y))

def get_ensemble(p): return np.mean(p, axis = 0)

# def get_mae(data): get_mean(data["prediction"], data["label"])

# def get_ensemble_mae(data)


k = 'PCE-LSTM NO discriminator'
k = "DeepConvLSTM"
k = "NOPCE-LSTM"
k = 'PCE-LSTM discriminator'
k = 'FFNN'

model_processed = dict()

for k in results.keys():

    nodisc = results[k]
    processed = dict()
    for idx in nodisc.keys():
        ind = nodisc[idx]
        first = ind[next(iter(ind.keys()))]

        if "_LSTM" in first.keys():

            first = first["_LSTM"]
            for k1,v1 in ind.items():
                ind[k1] = v1["_LSTM"]

        if "labels" not in first.keys():

            print("here")
            # first["labels"] = first["predictions"][0]
            # first["predictions"] = first["predictions"][1]
            for k1,v in ind.items():
                ind[k1]["labels"] = ind[k1]["predictions"][0].numpy()
                ind[k1]["predictions"] = ind[k1]["predictions"][1].numpy()
        label = first['labels'].reshape(-1)


        predictions  = np.stack([v["predictions"].reshape(-1) for v in ind.values()])

        ensemble = np.mean(predictions, axis=0)

        processed[idx+1] = {
            "label": label,
            "prediction": predictions,
        }
        model_processed[k] = processed

#%%
dmodel = model_processed["PCE-LSTM discriminator"]

ndmodel = model_processed["PCE-LSTM NO discriminator"]


def get_ind_mae(dmodel):
    r_metrics = dict()
    for k,ind in dmodel.items():
        mae = np.mean(np.abs(ind["prediction"] - ind["label"]),axis=1)
        r_metrics[k] = np.array(mae)
    return r_metrics

dr_metrics = get_ind_mae(dmodel)
ndr_metrics = get_ind_mae(ndmodel)

r_metrics[1], d_metrics[1]

import matplotlib.pyplot as plt
for i in d_metrics.keys():
    r = 
    plt.figure()
    plt.plot(d_metrics[i], dr_metrics[i]/ndr_metrics[i], '.')
    plt.show()

#%%

df =pd.DataFrame.from_dict(model_processed)

from collections import defaultdict
mae_dict = defaultdict(dict)

for ind, row in df.iterrows():
    label = row["NOPCE-LSTM"]['label'] 
    plt.figure()
    
    t = np.linspace(0, len(label)*2, len(label))
    plt.plot(t, label, label = "label")
    print(f"#### ind {ind} ####")
    for model in row.keys():
        if ("PCE" in model.upper()) and ( not 'PCE-LSTM discriminator' in model) :
            continue
        
        v = row[model]
        p = v["prediction"]
        e = get_ensemble(p)
        y = v["label"]
        mean_mae = get_mae(y, p )
        ensemble_mae = get_mae(y, e)
        mae_dict[model][ind] = (mean_mae, ensemble_mae)
        print(model, get_mae(y, p), get_mae(y,e))

        t = np.linspace(0, len(label)*2, len(e))
        plt.plot(t, e, label = model)
    plt.legend()
    plt.show()
#%%


# inds = [1,3,5,7]
inds = [1,2,3,4,5]

fig, axs = plt.subplots(len(inds), sharey=False, sharex=False, figsize=(10,len(inds)*5))
for idx, ind in enumerate(inds):
    ax = axs[idx]
    ax.set_title(f"Subject {ind}")    
    dcl = df["NOPCE-LSTM"].loc[ind]
    label = df["NOPCE-LSTM"].loc[ind]["label"]
    t = np.linspace(22, 2*len(label) + 22, len(label))

    axs[idx].plot(t, label, label="label")


    for model,legend in (("DeepConvLSTM", "DeepConvLSTM"),
                        ("FFNN", "FFNN"),
                        ("PCE-LSTM discriminator", "PCE-LSTM")): 
        data  = df[model].loc[ind]
        e = get_ensemble(data["prediction"])
        if model != "FFNN":
            t = np.linspace(22, 2*len(label) + 22, len(e))
        else:
            t = np.linspace(0, 2*len(label), len(e)) 
        
        axs[idx].plot(t,e, label = legend) 
    
    axs[idx].set_ylabel("Heart Rate [bpm]")
    axs[idx].legend()
axs[idx].set_xlabel("Time [seconds]")

plt.tight_layout()
fig.savefig("figures/pamap2_samples.pdf")




#%%

df_filtered = pd.DataFrame.from_dict(mae_dict)

#%%
def get_mean_mae(d): return get_mae(d["label"], d["prediction"])
def get_ensemble_mae(d): return  get_mae(d["label"], get_ensemble(d["prediction"]))




mae_mean = df.applymap(get_mean_mae)
mae_ensemble = df.applymap(get_ensemble_mae)

# df_mae_str = df_filtered.applymap(lambda x: f"{x[0]: .2f} vert {x[1]:.1f}")
df_mae = mae_ensemble.sort_index().transpose()
df_mae["average"] = df_mae.mean(axis = 1)
df_mae_str = df_mae.applymap(lambda x: f"{x:.2f}")
# print(df_mae_str.sort_index().transpose().to_latex().replace("vert", "$\\vert$"))
print(df_mae_str.to_latex())


# for k in sorted(processed.keys()):
#     v = processed[k]
#     p = v["prediction"]
#     e = get_ensemble(p)
#     y = v["label"]
#     print(k, get_mean(y, p), get_mean(y,e))
#     t = np.linspace(0, len(e)*2, len(e))
#     plt.figure()
#     plt.plot(t, y)
#     plt.plot(t, e)
#     plt.show()

# np.mean(np.abs(predictions-label)), np.mean(np.abs(ensemble-label))
# t = np.linspace(0,len(label)*2, len(label))



# for k in sorted(processed.keys()):
#     v = processed[k]
#     p = v["prediction"]
#     e = get_ensemble(p)
#     y = v["label"]
#     print(k, get_mean(y, p), get_mean(y,e))
#     t = np.linspace(0, len(e)*2, len(e))
#     plt.figure()
#     plt.plot(t, y)
#     plt.plot(t, e)
#     plt.show()

# np.mean(np.abs(predictions-label)), np.mean(np.abs(ensemble-label))
# t = np.linspace(0,len(label)*2, len(label))



# plt.plot(t, "label")







#%%

import numpy as np


diff =np.array( [
-0.06487695749,
-0.1014234875,
-0.1427003293,
-0.3677172875,
-0.0367731556,
0.1225953701,
-0.3157407407,
-0.5244680851,
0.1487854251,
-0.003571428571,
-0.008080808081,
-0.04481434059,
-0.02096436059,
-0.1689303905,
-0.06153846154
]).reshape(-1,1)

acc = np.array([0.55,
0.58,
0.6,
0.59,
0.67,
0.54,
0.57,
0.66,
0.56,
0.54,
0.6,
0.53,
0.51,
0.59,
0.6]).reshape(-1,1)

diff2 = np.array([
    0.2318718381,
0.05263157895,
-0.03210463734,
0.2518684604,
0.06152433425,
-0.1885964912,
-0.1060094531,
0.0584498094
]).reshape(-1,1)

acc2 = np.array([
    0.47,
0.5,
0.51,
0.51,
0.49,
0.57,
0.6,
0.54
]).reshape(-1,1)

import matplotlib.pyplot as plt
from sklearn import linear_model, metrics

m2 = linear_model.LinearRegression()
m2.fit(acc2, diff2)

m = linear_model.LinearRegression()
m.fit(acc, diff)


jacc = np.concatenate([acc, acc2])
jdiff = np.concatenate([diff, diff2])

p2 = m2.predict(jacc)#[*acc, *acc2])
p = m.predict(jacc)

mj = linear_model.LinearRegression()
mj.fit(jacc, jdiff)

jp = mj.predict(jacc)

print(metrics.r2_score(jdiff, jp))



plt.figure(figsize=[10,7])
plt.plot(acc, -diff, 'o', label="Dalia")
plt.plot(acc2, -diff2, '.', label="Pamap2")
plt.plot(jacc, -p2, '-')
plt.plot(jacc, -p, '-')
plt.plot(jacc, -jp, '-', label="Linear Regression")
plt.xlabel("PCE discrimination accuracy")
plt.ylabel("MAE relative reduction")
plt.legend()
plt.show()
plt.figure()
plt.plot(acc, diff, '.')
plt.figure()
plt.plot(acc2, diff2, '.')
plt.show()
# %%
