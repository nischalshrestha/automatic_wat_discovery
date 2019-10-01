#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
print(sklearn.__version__)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

df=pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
plt.show()


# In[ ]:


from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize=(18, 12));


# In[ ]:


df.columns[df.isna().any()]


# In[ ]:


df_train=df.copy()
drop_cols = ['PassengerId', 'Name', 'Ticket']
df_train.drop(drop_cols, axis=1, inplace=True)
y=df['Survived']
df_train.drop(['Survived'], axis=1, inplace=True)


# In[ ]:


cate_ids = []
num_ids = []
date_ids = []
dtypes = df_train.dtypes
for index, dtype in dtypes.iteritems():
    if dtype=='object':
        cate_ids.append(dtypes.index.get_loc(index))
    elif dtype.name.startswith('datetime'):
        date_ids.append(dtypes.index.get_loc(index))
    else:
        num_ids.append(dtypes.index.get_loc(index))
print(cate_ids, num_ids, date_ids)


# In[ ]:


# df_num = df_train.iloc[:, num_ids]
# df_cate = df_train.iloc[:, cate_ids].fillna('NAN')
# X_num = df_num.values
# X_cate = df_cate.values
# df_train=pd.get_dummies(df_train)
# embarked_oh_encoder = OneHotEncoder(categorical_features=[0])
# X_embarked = embarked_oh_encoder.fit_transform(X_cate[:, [2]]).toarray()[:, 1:]


# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_string_dtype, is_numeric_dtype
class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, ids = [], fillna = False, addna = False):
        self.fillna = fillna
        self.ids = ids
        self.addna = addna
        self.na_dict = {}
        self.imputers = {}
    def fit(self, X, y=None):
        for index, col in X.iloc[:, self.ids].items():
            if is_numeric_dtype(col) and self.fillna:
                self.imputers[index] = col.median()
                if self.addna and pd.isnull(col).sum():
                    self.na_dict[index] = True
        return self
    def transform(self, X, y=None):
        dfx = X.iloc[:, self.ids]
        for index, col in dfx.items():
            if is_numeric_dtype(col):
                if index in self.na_dict:
                    dfx[index+'_na'] = pd.isnull(col)
                if index in self.imputers:
                    col.fillna(self.imputers[index], inplace=True)
            elif self.fillna:
                col.fillna('NAN', inplace=True)
        
        return dfx


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler

num_pipeline = Pipeline([
    ('select_num', SelectFeatures(ids=num_ids, fillna='median', addna=True)),
])
X_num = num_pipeline.fit_transform(df_train)


# In[ ]:


class CatEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, max_n_cat=7, one_hot=False, orders={}):
        self.cats = {}
        self.one_hot = one_hot
        self.max_n_cat = max_n_cat
        self.orders = orders
    def fit(self, X, y=None):
        df_cat = X.copy()
        for n,c in df_cat.items():
            if is_string_dtype(c):
                df_cat[n] = c.astype('category').cat.as_ordered()
                if n in self.orders:
                    df_cat[n].cat.set_categories(self.orders[n], ordered=True, inplace=True)
                cats_count = len(df_cat[n].cat.categories)
                if not self.one_hot or (cats_count<=2 or cats_count>self.max_n_cat):
                    self.cats[n] = df_cat[n].cat.categories
        
        return self
    def transform(self, X, y=None):
        for n,c in X.items():
            if n in self.cats:
                X[n] = pd.Categorical(c, categories=self.cats[n], ordered=True)
                X[n] = X[n].cat.codes + 1
            else:
                X[n] = c.astype('category').cat.as_ordered()
        if self.one_hot:
            return pd.get_dummies(X, dummy_na=True)
        return X


# In[ ]:


cat_pipeline = Pipeline([
    ('select_cate', SelectFeatures(ids=cate_ids)),
    ('cat_encoder', CatEncoder())
])
X_cate = cat_pipeline.fit_transform(df_train)


# In[ ]:


df_train = pd.concat([X_num, X_cate], axis=1)
std_scaler = StandardScaler()
X_train_np = std_scaler.fit_transform(df_train)


# In[ ]:


from sklearn.metrics import mean_squared_error
def rmse(x,y):
    return np.sqrt(mean_squared_error(y, x))
#     return np.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)
          ]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

def keras_score(m, verbose=1):
    train_score = rmse(m.predict(X_train), y_train)
    valid_score = rmse(m.predict(X_valid), y_valid)
    if verbose:
        print(train_score, valid_score, m.evaluate(X_train, y_train), m.evaluate(X_valid, y_valid))
    return {'train_score': train_score, 'valid_score': valid_score}


# In[ ]:


from sklearn.model_selection import cross_val_score, KFold
cv = KFold(10, random_state=3)


# In[ ]:


from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(oob_score=True, random_state=3, n_jobs=-1, max_features=0.5)
params ={
    'n_estimators': [50, 60, 70, 80, 90, 100],
    'min_samples_leaf': [1, 3],# 5],# 10],# 25],
#     'max_features': [None, 0.5, 'sqrt', 'log2'],
    'max_depth': [8, 9, 10, 11],
    'min_samples_split': [2, 3, 4]
}

# best_score = 0
# for g in ParameterGrid(params):
#     model.set_params(**g)
#     model.fit(X_train_np, y)
#     if model.oob_score_ > best_score:
#         best_score = model.oob_score_
#         best_grid = g

# print("OOB score: %0.5f" % best_score)
# print("Grid param:", best_grid)


# In[ ]:


# best_params = {'max_features': 0.5, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
# rf_m = RandomForestClassifier(**best_params, random_state=3)
# score = cross_val_score(rf_m, X_train_np, y, cv=cv, n_jobs=-1)
# print(score.mean())


# In[ ]:


# rfc_m = RandomForestClassifier(**best_params, oob_score=True, random_state=3)
# rfc_m.fit(X_train_np, y)
# print(rfc_m.oob_score_)


# In[ ]:


from xgboost import XGBClassifier
xgb_model=XGBClassifier()
params ={
    'n_estimators': [100, 150, 200, 300],
    'learning_rate': [0.1, 0.05],
}
# grid = GridSearchCV(xgb_model, params, cv=cv)
# grid.fit(X_train_np, y)
# print(grid.best_params_)
# print(grid.best_score_)


# In[ ]:


best_params = {'learning_rate': 0.1, 'n_estimators': 200}
xg_m = XGBClassifier(**best_params, n_jobs=-1)
xg_m.fit(X_train_np, y)


# In[ ]:


def rf_feat_importance(m, df):
#     return sorted(zip(m.feature_importances_, df.columns), reverse=True)
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)
fi = rf_feat_importance(xg_m, df_train)
fi[:]


# In[ ]:


fi.plot('cols', 'imp', figsize=(15,8), legend=False);


# In[ ]:


to_keep = fi[fi.imp>0.02].cols
to_keep = [col for _, col in to_keep.items()]
df_keep = df_train[to_keep].copy()
std_scaler_keep = StandardScaler()
X_train_keep = std_scaler_keep.fit_transform(df_keep)


# In[ ]:


xg_m = XGBClassifier(**best_params)
score = cross_val_score(xg_m, X_train_keep, y, cv=cv, n_jobs=-1)
print(score.mean())


# In[ ]:


# rfc_m = RandomForestClassifier(**best_params, random_state=3)
# score = cross_val_score(rfc_m, X_train_keep, y, cv=cv, n_jobs=-1)
# # rfc_m.fit(X_train_keep, y)
# print(score.mean())


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings(action='once')


# In[ ]:


# from sklearn.model_selection import RandomizedSearchCV
# rndf_clf=RandomForestClassifier()
# params ={
#     'max_depth':list(range(3,10)),
#     'min_samples_leaf': list(range(4,10)),
#     'min_samples_split': list(range(5,30,2)),
#     'n_estimators': list(range(50, 300, 10))
# }
# rnd_search = RandomizedSearchCV(rndf_clf, params, cv=10, scoring='accuracy', n_iter=10, random_state=42)
# rnd_search.fit(X_train_np, y)
# print(grid_search.best_score_)
# print(grid_search.best_params_)


# In[ ]:


final_model = XGBClassifier(**best_params, n_jobs=-1)
# final_model = RandomForestClassifier(**best_params, n_jobs=-1)
final_model.fit(X_train_keep, y)


# In[ ]:


# df_final_test= pd.get_dummies(df_test)
# df_final_train, df_final_test = df_train.align(df_final_test, join='left', axis=1)
df_test=pd.read_csv('../input/test.csv')
df_t = df_test.drop(drop_cols, axis=1)
X_test_num = num_pipeline.transform(df_t)
X_test_cate = cat_pipeline.transform(df_t)
df_t = pd.concat([X_test_num, X_test_cate], axis=1)
X_test_keep = std_scaler_keep.transform(df_t[to_keep])


# In[ ]:


# final_model = rnd_search.best_estimator_
y_pred=final_model.predict(X_test_keep)
df_pred=pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_pred})
print(df_pred['Survived'].value_counts())
df_pred.to_csv('pred.csv', index = False)

