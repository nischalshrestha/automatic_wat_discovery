#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
print(os.listdir('../input'))


# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# lets check our train and test sets for any possible issues

# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.info()
print('-'*50)
test.info()


# it shows that Age and Cabin columns have some missing data

# In[ ]:


# survived
sns.countplot(train['Survived'])


# In[ ]:


train.columns


# In[ ]:


train.dtypes


# In[ ]:


list1=['Pclass',  'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'Cabin', 'Embarked']
for col in list1:
    g = sns.FacetGrid(train, col='Survived')
    g.map(plt.hist, col, bins=20)
    plt.suptitle(col)


# In[ ]:


# replace missing data with 0
train['Age'].replace({np.nan:0},inplace=True)
test['Age'].replace({np.nan:0},inplace=True)


# In[ ]:


# encoding data

from sklearn.preprocessing import LabelEncoder
#train.drop(['Name','Cabin'],axis=1)
list1=['Sex','Embarked','Ticket']
train[list1]=train[list1].astype('str').apply(LabelEncoder().fit_transform)


# In[ ]:


train_corr=train.corr()
plt.subplots(figsize=(8,8))
sns.heatmap(train_corr,annot=True,square=True)


# In[ ]:


Xtrain=train.drop(['PassengerId', 'Survived','Name','Cabin'],axis=1)
y=train['Survived']


# here we built a non-linear model to get the important variables by building extra trees model

# xgboost

# In[ ]:


cv_params={'max_depth':[3,5,7],'min_child_weight':[1,3,5]}
clf=xgb.XGBClassifier({'learning_rate':0.1,'n_estimators':1000,'seed':0,'subsample':0.8,'colsample_bytree':0.8,
           'objective':'binary:logistic'})
gs=GridSearchCV(estimator=clf,param_grid=cv_params,scoring='accuracy',cv=5,n_jobs=-1)
gs.fit(Xtrain,y)
print(gs.grid_scores_)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


# In[ ]:


#cv_params={'learning_rate':[0.1,0.01],'subsample':[0.7,0.8,0.9]}
#clf=xgb.XGBClassifier({'n_estimators':1000,'seed':0,'colsample_bytree':0.8,
#           'objective':'binary:logistic','max_depth':7, 'min_child_weight':3})
#gs=GridSearchCV(estimator=clf,param_grid=cv_params,scoring='accuracy',cv=5,n_jobs=-1)
#gs.fit(Xtrain,y)
#print('gs.grid_score_')
#print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))


#  **Early Stopping CV**
#  
#  To create a DMatrix:

# In[ ]:


xgb_params={
    'eta':0.5,
    'max_depth':7,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'objective':'binary:logistic',
    'silence':1,
    'seed':0,
    'min_child_weight':3
}
xgdmat=xgb.DMatrix(Xtrain,y,feature_names=Xtrain.columns.values)
model=xgb.train(dict(xgb_params,silent=0),xgdmat)


# In[ ]:


xgb.plot_importance(model)


# In[ ]:


test=test.drop(['Name','Cabin'],axis=1)
test_object_col=list(test.select_dtypes('object').columns.values)
test[test_object_col]=test[test_object_col].astype(str).apply(LabelEncoder().fit_transform)
xtest=test.drop(['PassengerId'],axis=1)


# In[ ]:


xtest=test.drop(['PassengerId'],axis=1)


# In[ ]:


testdmat=xgb.DMatrix(xtest)
from sklearn.metrics import accuracy_score
y_pred=model.predict(testdmat)
y_pred


# In[ ]:


y_pred[y_pred>0.5]=1
y_pred[y_pred<=0.5]=0
y_pred


# In[ ]:


submission=pd.DataFrame({'PassengerId':test['PassengerId'],'survived':y_pred})
submission['survived']=submission['survived'].astype(int)
submission.to_csv('submission.csv',index=False)


# In[ ]:




