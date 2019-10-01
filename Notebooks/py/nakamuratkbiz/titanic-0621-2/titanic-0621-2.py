#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[14]:


import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# # Load Dataset & Feature Engineering

# In[36]:


train_df = pd.read_csv("../input/train.csv")
train_df.head()


# In[37]:


train_df.info()


# In[38]:


train_df = train_df[train_df["Age"].isnull() == False]
train_df = train_df.iloc[:,[1,2,4,5,6,7,9]]
train_df = train_df.replace("female", 0).replace("male", 1)
train_df.head()


# In[39]:


import seaborn as sns

colormap = plt.cm.RdBu
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[40]:


train_df = train_df.iloc[:,[0,1,2,6]]
train_df.head()


# # Training & Evaluate

# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    train_df.iloc[:,1:],train_df.iloc[:,[0]], random_state=0)
X_train.head()


# In[42]:


from sklearn.preprocessing import MinMaxScaler

# preprocessing using 0-1 scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Logistic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression

# Training
logreg = LogisticRegression(C=1.0)
logreg.fit(X_train, y_train)

# Evaluation
print("Accuracy on training set: {:.2f}".format(logreg.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(logreg.score(X_test, y_test)))


# ## Random Forest

# In[44]:


from sklearn.ensemble import RandomForestClassifier

# Training
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

# Evaluation
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))

# 特徴量の重要度
n_features = X_train.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)


# ## Gradient Boosting

# In[45]:


from sklearn.ensemble import GradientBoostingClassifier

# Training
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

# Evaluation
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# 特徴量の重要度
n_features = X_train.shape[1]
plt.barh(range(n_features), gbrt.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n_features)



# ## Support Vector Machine

# In[46]:


from sklearn.svm import SVC

# Training
svc = SVC(C=1.0)

# Training
svc.fit(X_train_scaled, y_train)

# Evaluation
print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))


# ## Deep Learning

# In[47]:


from sklearn.neural_network import MLPClassifier

# Training
# mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[10,10])
mlp = MLPClassifier(solver='lbfgs')
mlp.fit(X_train_scaled, y_train)
        
# Evaluation
print("Accuracy on training set: {:.2f}".format(mlp.score(X_train_scaled, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score(X_test_scaled, y_test)))


# # Tune

# In[49]:


# 勾配ブースティング)
from sklearn.ensemble import GradientBoostingClassifier
## 交差検証
from sklearn.model_selection import cross_val_score
## 層化k分割交差検証(グループ付きにする場合は、GroupKFoldを使用する)
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True)

# Best Parameterの検出
result = pd.DataFrame(index=[],columns=["depth","n_est","l_rate","score"])
best_score = 0
index = 0
for max_depth in [2,3,4]:
    for n_estimators in [100, 150, 200, 300, 400]:
        for learning_rate in [0.01, 0.025, 0.05, 0.1, 0.2]:
            gbrt = GradientBoostingClassifier(
                max_depth = max_depth, 
                n_estimators = n_estimators,
                learning_rate = learning_rate,
                random_state = 0)

            scores = cross_val_score(gbrt, X_train, y_train, cv=kfold)
            scores = scores.mean()
            
            result.loc[index] = [max_depth, n_estimators, learning_rate, scores]
            index += 1
            if scores > best_score:
                best_score = scores
                best_parameters = {
                    "max_depth":max_depth,
                    "n_estimators":n_estimators,
                    "learning_rate":learning_rate
                    }
# Training
gbrt = GradientBoostingClassifier(**best_parameters)
gbrt.fit(X_train, y_train)

# Evaluation
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))


# In[50]:


gbrt


# # Predict

# In[64]:


test_df = pd.read_csv("../input/test.csv")
test_df.head()


# In[65]:


test_df.info()


# In[66]:


test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean())
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].mean())


# In[68]:


test_df = test_df.replace("female", 0).replace("male", 1)
test_df.head()


# In[69]:


test_df = test_df.iloc[:,[0,1,3,8]]
test_df.head()


# In[70]:


pdict = gbrt.predict(test_df.iloc[:,1:])


# In[71]:


result_df = pd.DataFrame(test_df.iloc[:,0], columns=["PassengerId"])


# In[72]:


result_df["Survived"] = pdict


# In[73]:


result_df.head()


# In[75]:


result_df.to_csv("result_0621_2.csv", index=None)


# In[76]:


print(os.listdir("."))


# In[ ]:




