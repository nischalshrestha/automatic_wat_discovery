#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob


# In[ ]:


def import_ds(path):
    Ts = pd.read_csv(path, index_col='PassengerId')
    
    #Feature engineering
    Ts["Sex"] = Ts["Sex"].apply(lambda x: 1 if x=="female" else 0 if x=="male" else 0.5)
    
    #feature extracting
    ts = Ts[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    
    # Normalizing
    ts=(ts-ts.min())/(ts.max()-ts.min())
    
    
    if "Survived" in Ts.columns:
        ts.loc[:, "Survived"] = Ts["Survived"]
    else:
        ts.loc[:, "Survived"] = 0
    
    # Add bias(
    ts["Bias"] = 1
    
    return ts


# In[ ]:


ts = import_ds("../input/train.csv")


# In[ ]:





# In[ ]:


ts = ts.dropna(axis=0, how='any')
y = np.array(ts["Survived"]).reshape(ts.shape[0], 1)
ts.drop(["Survived"], axis=1, inplace=True)
m = ts.shape[0]
n = ts.shape[1]


# In[ ]:


theta = np.random.normal(size=(n, 1))
lambda_p = 1
steps = 1000
costs = np.zeros(steps)
sigmoid = np.vectorize(lambda x:1/(1+np.exp(-x)))
for i in range(steps):
    hip = sigmoid(ts.as_matrix().dot(theta))
    costs_p = np.vectorize(lambda x: np.log(x) if x > 0 else np.log(1+x))(np.multiply(2*y-1, hip))
    costs[i] = -np.sum(costs_p)/m
    grad = (hip - y).T.dot(ts.as_matrix()).T/m
    theta -= lambda_p * grad


# In[ ]:


df = import_ds("../input/test.csv").drop(["Survived"], axis=1)


# In[ ]:


thresh = 0.5
df.loc[:, "Survived"] = np.vectorize(lambda x: 1 if x > thresh else 0)(sigmoid(df.as_matrix().dot(theta)))


# In[ ]:


df["Survived"].to_frame().reset_index().to_csv("prev.csv", index=None)


# In[ ]:





# In[ ]:




