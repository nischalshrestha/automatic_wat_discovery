#!/usr/bin/env python
# coding: utf-8

# In[6]:


# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
#load data
train_df=pd.read_csv('../input/train.csv',header=0)
test_df=pd.read_csv('../input/test.csv',header=0)


# We'll impute missing values using the median for numeric columns and the most
# common value for string columns.
# This is based on some nice code by 'sveitser' at http://stackoverflow.com/a/25562948
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self,X,y=None):
        self.fill=pd.Series([X[c].value_counts().index[0]
            if X[c].dtype==np.dtype('O') else X[c].median() for c in X],index=X.columns)
        return self
        
    def transform(self,X,y=None):
        return X.fillna(self.fill)
        
feature_columns_to_use=['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns=['Sex']


# Join the features from train and test together before imputing missing values,
# in case their distribution is slightly different

big_X=train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed=DataFrameImputer().fit_transform(big_X)

# XGBoost doesn't (yet) handle categorical features automatically, so we need to change
# them to columns of integer values.
# See http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing for more
# details and options
le=LabelEncoder()
for feature in nonnumeric_columns:
    big_X_imputed[feature]=le.fit_transform(big_X_imputed[feature])
train_X=big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X=big_X_imputed[train_df.shape[0]::].as_matrix()
train_y=train_df['Survived']

# You can experiment with many other options here, using the same .fit() and .predict()
# methods; see http://scikit-learn.org
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost
gbm =xgb.XGBClassifier(max_depth=3,n_eatimators=300,learning_rate=0.05).fit(train_X,train_y)
predictions=gbm.predict(test_X)
print(predictions)


submission=pd.DataFrame({'PassengerId':test_df['PassengerId'],'Survived':predictions})
submission.to_csv("submission.csv",index=False)


# In[ ]:




