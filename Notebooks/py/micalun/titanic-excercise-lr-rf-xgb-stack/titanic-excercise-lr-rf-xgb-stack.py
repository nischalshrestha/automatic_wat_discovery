#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df_train_index =df_train.index.size


# In[ ]:


df_train.head()


# In[ ]:


df_all = pd.concat([df_train,df_test],ignore_index=True)


# In[ ]:


df_all.head(10)


# In[ ]:


null_count = df_all.isnull().sum()


# In[ ]:


null_percent = 100*df_all.isnull().sum()/len(df_all)


# In[ ]:


pd.concat([null_count,null_percent],axis=1)


# In[ ]:


df_all.loc[df_all['Age'].isnull(),'Age']=df_all['Age'].mean()


# In[ ]:


df_all.loc[df_all['Fare'].isnull()]


# In[ ]:


fare_fill=df_all.loc[df_all['Pclass']==3].mean()['Fare']


# In[ ]:


df_all['Fare'].fillna(value=fare_fill,inplace=True)


# In[ ]:


df_all['Embarked'].unique()


# In[ ]:


le = LabelEncoder().fit(df_all['Sex'])
df_all['Sex'] = le.transform(df_all['Sex'])


# In[ ]:


df_all.columns


# In[ ]:


use_columns = ['Age', 'Embarked', 'Fare', 'Parch',
       'Pclass', 'Sex', 'SibSp', 'Survived']


# In[ ]:


df_all = pd.get_dummies(df_all[use_columns])


# In[ ]:


numerical_col = ['Age','Fare']
numerical_col_means = df_all.loc[:,numerical_col].mean()
numerical_col_stds = df_all.loc[:,numerical_col].std()
df_all.loc[:,numerical_col] = (df_all.loc[:,numerical_col]-numerical_col_means)/numerical_col_stds


# In[ ]:


df_all.head(10)


# In[ ]:


data_train = df_all.iloc[:891,:]


# In[ ]:


data_test = df_all.iloc[891:,:]


# In[ ]:


df_all.dtypes


# In[ ]:


y_train = data_train['Survived']
x_train = data_train.drop(columns='Survived')
x_test = data_test.drop(columns='Survived')


# In[ ]:


x_train.isnull().sum()


# In[ ]:


import xgboost as xgb
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV


# In[ ]:


c=np.logspace(-3,2,50)
param = {'C':c,'max_iter':[50,100,200]}
lr = LogisticRegression()
lr_gscv = GridSearchCV(lr,param_grid=param,cv=5,n_jobs=-1)
lr_gscv.fit(x_train,y_train)


# In[ ]:


lr_gscv.best_estimator_


# In[ ]:


lr_gscv.best_score_


# In[ ]:


y_hat_lr = lr_gscv.predict(x_test)


# In[ ]:


param_rfc = { 'max_features':np.linspace(0.1,1,10),'min_samples_split':[2,3,4,5]}
rfc = GridSearchCV(RandomForestClassifier(n_estimators=100),param_grid=param_rfc,cv=5,scoring='f1',n_jobs=-1)
rfc.fit(x_train, y_train)


# In[ ]:


rfc.best_estimator_


# In[ ]:


rfc.best_params_


# In[ ]:


y_train_hat=rfc.predict(x_train)


# In[ ]:


acc = (y_train_hat==y_train).sum()


# In[ ]:


acc/890


# In[ ]:


get_ipython().magic(u'pinfo2 xgb.XGBClassifier')


# In[ ]:


data_train = xgb.DMatrix(x_train, label=y_train)
# data_test = xgb.DMatrix(x_test)
watch_list = [(data_test, 'eval'), (data_train, 'train')]
param = { 'learning_rate':np.logspace(-3,1,10),'max_depth':[2,4,6,10]}
         # 'subsample': 1, 'alpha': 0, 'lambda': 0, 'min_child_weight': 1}
xgb_reg = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic',n_estimators=200),param_grid=param,cv=5)
bst = xgb_reg.fit(x_train,y_train)


# In[ ]:


xgb_reg.best_params_


# In[ ]:


xgb_reg.best_score_


# In[ ]:


y_hat = xgb_reg.predict(x_test)
# write_result(bst, 3)
y_hat[y_hat > 0.5] = 1
y_hat[~(y_hat > 0.5)] = 0
# xgb_rate = show_accuracy(y_hat, y_test, 'XGBoost ')


# In[ ]:


y_rfc_hat = rfc.predict(x_test)


# In[ ]:


df_rfc = pd.DataFrame(data={'PassengerId':df_test['PassengerId'].values,'Survived':y_rfc_hat},dtype=int)


# In[ ]:


df_rfc.to_csv('submission.csv',index=False)


# In[ ]:




