#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[ ]:


data_path = '../input'

def fetch_data(data_path):
    train_df = pd.read_csv(data_path + '/train.csv')
    test_df = pd.read_csv(data_path + '/test.csv')
    samp_df = pd.read_csv(data_path + '/gender_submission.csv')
    return train_df, test_df, samp_df


# In[ ]:


train_df, test_df, samp_df = fetch_data(data_path)


# ## Data exploration

# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.Survived.value_counts()


# In[ ]:


train_df.Pclass.value_counts()


# In[ ]:


train_df.Sex.value_counts()


# In[ ]:


train_df.Embarked.value_counts()


# ## Feature engineering

# In[ ]:


corr_matrix = train_df.corr()
corr_matrix['Survived'].sort_values(ascending=False)


# In[ ]:


train_df['Family_number'] = train_df.SibSp + train_df.Parch # number of family members on board
train_df['Age_cats'] = np.ceil(train_df.Age / 18) # Creating Age ranges to split into categories


# In[ ]:


corr_matrix = train_df.corr()
corr_matrix['Survived'].sort_values(ascending=False)


# In[ ]:


train_df.head()


# ## Data preparation

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attrib_names):
        self.attrib_names = attrib_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attrib_names]


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare", "Family_number"])),
        ("imputer", Imputer(strategy="median")),
    ])


# In[ ]:


num_pipeline.fit_transform(train_df)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import CategoricalImputer


# In[ ]:


class MostFrequentImputer(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        self.imputer = CategoricalImputer()
        return self
    def transform(self,X):
        age_cats_imputed = pd.Series(self.imputer.fit_transform(X.Age_cats.copy())).astype('category')
        sex_imputed = pd.Series(self.imputer.fit_transform(X.Sex.copy())).astype('category')
        embarked_imputed = pd.Series(self.imputer.fit_transform(X.Embarked.copy())).astype('category')
        X.Sex = sex_imputed.cat.codes
        X.Embarked = embarked_imputed.cat.codes
        X.Age_cats = age_cats_imputed
        return X


# In[ ]:


cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked", "Age_cats"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])


# In[ ]:


cat_pipeline.fit_transform(train_df)


# In[ ]:


from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[ ]:


x_train = preprocess_pipeline.fit_transform(train_df)
x_train


# In[ ]:


y_train = train_df["Survived"]


# In[ ]:


test_df['Family_number'] = test_df.SibSp + test_df.Parch # number of family members on board
test_df['Age_cats'] = np.ceil(test_df.Age / 18) # Creating Age ranges to split into categories


# ## Run on models

# In[ ]:


# SVM
from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(x_train,y_train)


# In[ ]:


X_test = preprocess_pipeline.transform(test_df)
y_pred = svm_clf.predict(X_test)


# In[ ]:


from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, x_train, y_train, cv=10)
svm_scores.mean()


# In[ ]:


# Random forest
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(x_train,y_train)
forest_scores = cross_val_score(forest_clf, x_train, y_train, cv=10)
forest_scores.mean()


# In[ ]:


plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 20, 30], 'max_features': [2, 4, 6, 8, 10, 12]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10, 20], 'max_features': [2, 3, 4]},
]

forest_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='accuracy', return_train_score=True)
grid_search.fit(x_train,y_train)


# In[ ]:


grid_search.best_params_


# In[ ]:


grid_search.best_estimator_


# In[ ]:


cvres = grid_search.cv_results_
for acc_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(acc_score, params)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_clf = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(forest_clf, param_distributions=param_distribs,
                                n_iter=30, cv=5, scoring='accuracy', random_state=42)
rnd_search.fit(x_train, y_train)


# In[ ]:


cvres = rnd_search.cv_results_
for acc_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(acc_score, params)


# In[ ]:


rnd_search.best_score_


# In[ ]:


rnd_search.best_estimator_


# ## Generate test predictions

# In[ ]:


preds = rnd_search.best_estimator_.predict(X_test)
preds


# In[ ]:


preds.shape


# In[ ]:


samp_df.head()


# In[ ]:


subm_df = samp_df.copy()


# In[ ]:


subm_df.Survived = preds
subm_df.head()


# In[ ]:


subm_df.to_csv('submission.csv')

