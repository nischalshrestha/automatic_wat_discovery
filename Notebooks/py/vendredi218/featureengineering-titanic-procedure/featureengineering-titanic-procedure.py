#!/usr/bin/env python
# coding: utf-8

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import sklearn
import scipy as sp

import random
import sys
import time

import IPython
from IPython import display

import warnings
warnings.filterwarnings('ignore')
print('-'*25)

import os
print(os.listdir("../input"))
print('-'*25)


# In[2]:


# common model algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import lightgbm as lgb

#common model helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#configure visualization defaults
get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[3]:


train_raw = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df = train_raw.copy(deep=True)
df_data = pd.concat([train_df, test_df])

df_data.sample(5)


# # 1.EDA

# # 2. 特征工程

# ## 2.1 completing

# In[4]:


# for dataset in data_cleaner:
#     for i in ['male','female']:
#         for j in range(0,3):
#             dataset.loc[(dataset.Age.isnull()) & (dataset.Sex==i) & (dataset.Pclass==j+1), 'Age']=dataset.loc[
#                  (dataset.Sex==i) & (dataset.Pclass==j+1), 'Age'].median()
#     dataset['Age'] = dataset['Age'].astype(int)
df_data['Embarked'].fillna(df_data['Embarked'].mode()[0], inplace=True)
df_data['Fare'].fillna(df_data['Fare'].median(), inplace = True)
df_data['Cabin'] = df_data['Cabin'].apply(lambda x:x[0] if x is not np.nan else 'X')
cabin_counts = df_data['Cabin'].value_counts()
df_data['Cabin'] = df_data['Cabin'].apply((lambda x:'X' if cabin_counts[x] < 10 else x))


# In[5]:


def predict_age1(dataset):
    data_p = dataset[['Pclass','SibSp','Parch','Fare','Age']]
    x_train = data_p.loc[~data_p['Age'].isnull(), :].drop('Age', 1)
    y_train = data_p.loc[~data_p['Age'].isnull(), :]['Age']
    x_test = data_p.loc[data_p['Age'].isnull(), :].drop('Age', 1)
    
    rfr = ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x_train,y_train)
    
    y_test = rfr.predict(x_test)
    return y_test

df_data.loc[df_data['Age'].isnull(), 'Age'] = predict_age1(df_data)


# In[ ]:


def predict_age(dataset):
    '''
    预测年龄
    '''
    data_p = dataset[['Pclass','SibSp','Parch','Fare','Age']]
    x_train = data_p.loc[~data_p['Age'].isnull(), :].drop('Age', 1)
    y_train = data_p.loc[~data_p['Age'].isnull(), :]['Age']
    x_test = data_p.loc[data_p['Age'].isnull(), :].drop('Age', 1)
    print('初步处理完')
    param_grid = {
        'learning_rate':[.001, .005, .01, .05, .1],
        'max_depth':[2, 4, 6, 8],
        'n_estimators':[50, 100, 300, 500, 1000],
        'seed':[2018]
    }
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0) 
    tune_model = model_selection.GridSearchCV(XGBRegressor(nthread=-1), param_grid=param_grid, 
                                              scoring = 'neg_mean_squared_error', cv = cv_split)
    print('model tuned')
    tune_model.fit(x_train, y_train)
    print('model fitted')
    print(tune_model.best_params_)
    y_test = tune_model.best_estimator_.predict(x_test)
    print('model predicted')
    print(y_test.head(5))
    return y_test

# df_data.loc[df_data['Age'].isnull(), 'Age'] = predict_age(df_data)


# In[6]:


print(df_data.isnull().sum())
print("-"*10)


# ## 2.2 构造新特征 creating

# In[7]:


df_data['FamilySize'] = df_data['SibSp'] + df_data['Parch'] + 1
df_data['IsAlone'] = 1
df_data['IsAlone'].loc[df_data['FamilySize'] > 1] = 0

df_data['Title'] = df_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_counts = df_data['Title'].value_counts()
df_data['Title'] = list(map(lambda x:'Rare' if title_counts[x] < 10 else x, df_data['Title']))


# In[8]:


df_data['Family_Name'] = df_data['Name'].apply(lambda x: str.split(x, ",")[0])

DEFAULT_SURVIVAL_VALUE = 0.5
df_data['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in df_data.groupby(['Family_Name', 'Fare']):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in df_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    df_data.loc[df_data['PassengerId'] == passID, 'Family_Survival'] = 0


# In[9]:


df_data['FareBin'] = pd.qcut(df_data['Fare'], 4)
df_data['AgeBin'] = pd.cut(df_data['Age'].astype(int), 5)


# ## 2.3 特征转换 convert

# LabelEncoder()和get_dummies转换分类变量

# In[13]:


label = LabelEncoder()
df_data['Sex_Code'] = label.fit_transform(df_data['Sex'])  # female为0, male为1
df_data['AgeBin_Code'] = label.fit_transform(df_data['AgeBin'])
df_data['FareBin_Code'] = label.fit_transform(df_data['FareBin'])
df_data = pd.concat([df_data, pd.get_dummies(df_data[['Embarked', 'Title', 'Cabin']])], axis=1)


# 特征因子化

# In[15]:


import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df_data['Age'].reshape(-1,1))
df_data['Age_scaled'] = scaler.fit_transform(df_data['Age'].reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df_data['Fare'].reshape(-1,1))
df_data['Fare_scaled'] = scaler.fit_transform(df_data['Fare'].reshape(-1,1), fare_scale_param)
df_data.info()


# ## 2.4 剔除特征 correcting

# In[16]:


df_data.info()


# In[17]:


drop_columns = ['Sex', 'Name','Cabin','Embarked','FareBin','AgeBin', 'Ticket', 'Title', 'Family_Name']
df_data = df_data.drop(drop_columns, 1)
df_data.to_csv('fe_rfrage_scaled_data.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




