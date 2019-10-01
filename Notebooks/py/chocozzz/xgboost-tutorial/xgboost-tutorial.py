#!/usr/bin/env python
# coding: utf-8

# XGBOOST 공식 문서 : https://xgboost.readthedocs.io/en/latest/

# In[ ]:


import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# Load in the train and test datasets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

all = pd.concat([ train, test ],sort=False)


# In[ ]:


all.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
all["Embarked"] = le.fit_transform(all["Embarked"].fillna('0'))
all["Sex"] = le.fit_transform(all["Sex"].fillna('3'))
all.head()


# In[ ]:


# split the data back into train and test
df_train = all.loc[all['Survived'].isin([np.nan]) == False]
df_test  = all.loc[all['Survived'].isin([np.nan]) == True]


# In[ ]:


print(df_train.shape)
print(df_test.shape)
df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


'''
# XGBoost model + parameter tuning with GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

feature_names = ['Sex','Embarked','Pclass','Survived','Age','SibSp','Parch','Fare']

xgb = XGBRegressor()
params={
    'max_depth': [2,3,4,5], 
    'subsample': [0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree': [0.5,0.6,0.7,0.8],
    'n_estimators': [1000,2000,3000],
    'reg_alpha': [0.01, 0.02, 0.03, 0.04]
}

grs = GridSearchCV(xgb, param_grid=params, cv = 10, n_jobs=4, verbose=2)
grs.fit(np.array(df_train[feature_names]), np.array(df_train['Survived']))

print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data: {0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))
# TODO: why is this so bad?
'''


# 위에는 그리드서치라는 기법을 이용해서 예측한건데 여기서 선택지가 3개로 갈림.
# 1. 그리드 서치   2. 랜덤 서치    3.  베이지안 최적화 
# 
# 그리드 < 랜덤 < 베이지안 으로 괜찮은 방법입니다.
# 
# Best parameters {'colsample_bytree': 0.8, 'max_depth': 2, 'n_estimators': 3000, 'reg_alpha': 0.01, 'subsample': 0.6}

# In[ ]:


train_y = df_train['Survived']; train_x = df_train.drop('Survived',axis=1)
excluded_feats = ['PassengerId','Ticket','Cabin','Name']
features = [f_ for f_ in train_x.columns if f_ not in excluded_feats]
features


# In[ ]:


from sklearn.model_selection import KFold
folds = KFold(n_splits=4, shuffle=True, random_state=546789)


# In[ ]:


oof_preds = np.zeros(train_y.shape[0])
sub_preds = np.zeros(df_test.shape[0])


# 아래는 XGBOOST 모델 만드는 부분인데 정보손실 최소화 하기 위해서 kfold방식 이용했습니다. 
# 
# 순서는 
# 
# 1. X변수와 Y변수를 분리해준다.
# 
# ```
#     trn_x, trn_y = train_x[features].iloc[trn_idx], train_y.iloc[trn_idx]
# 
#     val_x, val_y = train_x[features].iloc[val_idx], train_y.iloc[val_idx]
# ```
# 
# 2.  모델의 하이퍼파라미터를 조정해준다. 여기에서 위에서 말한 그리드서치, 랜덤서치, 베이지안 최적화가 쓰입니다.
# 
# 베이지안 최적화 in XGBOOST : http://www.kwangsiklee.com/2018/06/bayesianoptimization%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%98%EC%97%AC-xgboost-%ED%95%98%EC%9D%B4%ED%8D%BC-%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0%EB%A5%BC-%EC%B5%9C%EC%A0%81%ED%99%94/
# 
# ```
#     clf = XGBClassifier(
# 
#         objective = 'binary:logistic', 
# 
#         booster = "gbtree",
# 
#         eval_metric = 'auc', 
# 
#         nthread = 4,
# 
#         eta = 0.05,
# 
#         gamma = 0,
# 
#         max_depth = 2, 
# 
#         subsample = 0.6, 
# 
#         colsample_bytree = 0.8, 
# 
#         colsample_bylevel = 0.675,
# 
#         min_child_weight = 22,
# 
#         alpha = 0,
# 
#         random_state = 42, 
# 
#         nrounds = 2000,
#         
#         n_estimators=3000
# 
#     )
# ```   
#  3.   모델을 적용시킵니다.
#  
# ```
#     clf.fit(trn_x, trn_y, eval_set= [(trn_x, trn_y), (val_x, val_y)], verbose=10, early_stopping_rounds=100)
# ```
# 
# 2 번의 최적화 관련해서는 저기에 적힌것보다 많은 하이퍼 파라미터가 있는데 
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 이 링크에 설명이 잘 나와있습니다. 
# 그리고 모든 변수에 대한 설명은 https://xgboost.readthedocs.io/en/latest/parameter.html 링크에 나와있습니다.
# 
# 개인적으로 사람들이 많이 다루는건 **`booster`** 와 `eta` , `gmmama`,  **`max_depth`**,  `min_child_weight`, `subsample`, `colsample_bytree`, `scale_pos_weigh`, `max_leaves`를 많이 설정하는것 같았습니다.
# 
# 
# 

# In[ ]:


import gc
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_x)):

    trn_x, trn_y = train_x[features].iloc[trn_idx], train_y.iloc[trn_idx]

    val_x, val_y = train_x[features].iloc[val_idx], train_y.iloc[val_idx]

    

    clf = XGBClassifier(

        objective = 'binary:logistic', 

        booster = "gbtree",

        eval_metric = 'auc', 

        nthread = 4,

        eta = 0.05,

        gamma = 0,

        max_depth = 2, 

        subsample = 0.6, 

        colsample_bytree = 0.8, 

        colsample_bylevel = 0.675,

        min_child_weight = 22,

        alpha = 0,

        random_state = 42, 

        nrounds = 2000,
        
        n_estimators=3000

    )



    clf.fit(trn_x, trn_y, eval_set= [(trn_x, trn_y), (val_x, val_y)], verbose=10, early_stopping_rounds=100)

    

    oof_preds[val_idx] = clf.predict_proba(val_x)[:, 1]

    sub_preds += clf.predict_proba(df_test[features])[:, 1] / folds.n_splits

    

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))

    del clf, trn_x, trn_y, val_x, val_y

    gc.collect()

    

print('Full AUC score %.6f' % roc_auc_score(train_y, oof_preds))   



test['Survived'] = sub_preds



test[['PassengerId', 'Survived']].to_csv('xgb_submission_esi.csv', index=False, float_format='%.8f')

