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
