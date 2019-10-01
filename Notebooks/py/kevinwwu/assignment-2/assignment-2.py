#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

import os
print(os.listdir("../input"))


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


list(train)


# In[ ]:


features = list(test)
response = list(set(list(train)) - set(list(test)))


# In[ ]:


all_data = pd.concat((train[features],
                      test[features]))


# In[ ]:


all_data.isnull().sum()


# In[ ]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (18.0, 9.0)
all_data[numeric_feats].plot(kind='density', subplots=True, layout=(3,2), sharex=False)
plt.show()


# In[ ]:


all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median())
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (18.0, 9.0)
all_data[numeric_feats].plot(kind='density', subplots=True, layout=(3,2), sharex=False)
plt.show()


# In[ ]:


features.remove('PassengerId')
features.remove('Name')
features.remove('Ticket')
features.remove('Cabin')
features.remove('Embarked')


# In[ ]:


features


# In[ ]:


all_data['Pclass'] = all_data['Pclass'].astype('category')


# In[ ]:


all_data = pd.get_dummies(all_data)
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train[response]


# # Logistic

# In[ ]:


log_model = LogisticRegression()
log_model.fit(X_train, y)
log_pred = log_model.predict(X_test)
log_accuracy = log_model.score(X_train, y)


# In[ ]:


log_accuracy


# # Decision Tree Classifiers

# In[ ]:


tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X_train, y)
tree_pred = tree_model.predict(X_test)
tree_accuracy = tree_model.score(X_train, y)


# In[ ]:


tree_accuracy


# # Random Forest

# In[ ]:


forest_model = RandomForestClassifier()
forest_model.fit(X_train, y)
forest_pred = forest_model.predict(X_test)
forest_accuracy = forest_model.score(X_train, y)


# In[ ]:


forest_accuracy


# # XGBoost

# In[ ]:


xgb_model = XGBClassifier()
xgb_model.fit(X_train, y)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = xgb_model.score(X_train, y)


# In[ ]:


xgb_accuracy


# # Neural Network

# In[ ]:


# Build model
nn_model = Sequential()
BatchNormalization()
# layers
nn_model.add(Dense(90,input_dim=X_train.shape[1],activation='relu'))
BatchNormalization()
Dropout(0.5)

nn_model.add(Dense(30,activation='relu'))
BatchNormalization()
Dropout(0.5)

nn_model.add(Dense(1,activation='sigmoid'))
BatchNormalization()

# Compiling
nn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train
nn_model.fit(X_train, y, batch_size = 32, verbose=20)

nn_pred = nn_model.predict(X_test)

nn_pred = (nn_pred > 0.5).astype(int).reshape(X_test.shape[0])

nn_accuracy = nn_model.evaluate(X_train, y)


# In[ ]:


nn_accuracy


# In[ ]:




