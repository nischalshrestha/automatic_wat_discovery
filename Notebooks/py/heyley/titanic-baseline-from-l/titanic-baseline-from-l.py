#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os

import csv as csv
import sklearn as scl

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt

import seaborn as sns


# # Exploratory Data Analysis

# In[ ]:


train = pd.read_csv('../input/train.csv') 
train.shape


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


test = pd.read_csv('../input/test.csv') 
test.shape


# In[ ]:


test.describe()


# In[ ]:


features = list(set(test.columns) - {'Name'})
features


# In[ ]:


# Impute missing values using the median for numeric columns and the most common value for string columns.
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[ ]:


# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different
common = train[features].append(test[features])
common = DataFrameImputer().fit_transform(common)
common.describe()


# In[ ]:


# Change categorical features  to columns of integer values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feature in features:
    common[feature] = le.fit_transform(common[feature])


# In[ ]:


train2 = common[0:train.shape[0]]
test2 = common[train.shape[0]::]


# In[ ]:


correlation_matrix = train2.corr()
correlation_matrix


# In[ ]:


plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(correlation_matrix)


# In[ ]:


features2 = ['Sex','Age','Pclass','Parch','Embarked', 'Cabin']
train3 = train2[features2]
test3 = test2[features2]
correlation_matrix2 = train3.corr()
correlation_matrix2


# In[ ]:


plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(correlation_matrix2)


# In[ ]:


train_X = train3.values
test_X = test3.values
train_y = train['Survived']


# # Split dataset

# In[ ]:


from sklearn.utils import shuffle

df_train = train3.join(pd.DataFrame({'Survived': train_y}))
size = int(df_train.shape[0] * 0.8)
print(size)
train4 = shuffle(df_train)
train_80 = train4[:size]
train_20 = train4[size:]


# In[ ]:


cv_train_X = train_80[features2].values
cv_test_X = train_20[features2].values
cv_train_y = train_80['Survived']
cv_test_y = train_20['Survived']


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_X, train_y)
print(model)


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred = model.predict(train_X)
accuracy_score(train_y, y_pred)


# In[ ]:


# Cross-validation
model2 = LogisticRegression()
model2.fit(cv_train_X, cv_train_y)
y_pred = model2.predict(cv_test_X)
accuracy_score(cv_test_y, y_pred)


# In[ ]:


# Make predictions
preds = model.predict(test_X)


# In[ ]:


submission = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                            'Survived': preds })
submission.to_csv('submission.csv', index=False)


# # XGBoost

# In[ ]:


import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)


# In[ ]:


y_pred = gbm.predict(train_X)
accuracy_score(train_y, y_pred)


# In[ ]:


# Cross-validation
gbm2 = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(cv_train_X, cv_train_y)
y_pred = gbm2.predict(cv_test_X)
accuracy_score(cv_test_y, y_pred)


# In[ ]:


preds2 = gbm.predict(test_X)


# In[ ]:


submission2 = pd.DataFrame({ 'PassengerId': test['PassengerId'],
                             'Survived': preds2 })
submission2.to_csv('submission2.csv', index=False)

