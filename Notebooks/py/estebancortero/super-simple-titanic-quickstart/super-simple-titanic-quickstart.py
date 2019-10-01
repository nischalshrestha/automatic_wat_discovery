#!/usr/bin/env python
# coding: utf-8

# # Super Simple Titanic Quickstart
# 
# A super short, fast, and minimalistic notebook that compares a few different models using cross-validation. Finally, chooses the best (logistic regression).
# 
# Uses only four features: Pclass, Age, Sex, Familysize (based on analysis insights from other notebooks, e.g., [this outstanding notebook](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling/notebook)). No feature engineering, no hyper-paramter tuning, no fancy stuff but already surprisingly good.
# 
# **If you liked the notebook or have feedback, I'd appreciate upvotes and any comments!**
# 
# Edit: Added new CatBoost regressor, which - contrary to their claim - seems to take much longer than XGBoost and still perform worse (at least here, in terms of MAE with default parameters).

# In[16]:


# imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# read data, split into train, validate, test
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[17]:


# prepare data (outside of pipeline)
# convert sex into binary male feature
train_df["Male"] = train_df.Sex.map({"male": 1, "female": 0}).astype(int)
test_df["Male"] = test_df.Sex.map({"male": 1, "female": 0}).astype(int)

# calculate family size based on siblings and parents
train_df["FamilySize"] = train_df.SibSp + train_df.Parch + 1
test_df["FamilySize"] = test_df.SibSp + train_df.Parch + 1

# drop "unnecessary features" (keep id in test set)
train_df = train_df.drop(["PassengerId", "Ticket", "Cabin", "Name", "Sex", "SibSp", "Parch", "Embarked", "Fare"], axis=1)
test_df = test_df.drop(["Ticket", "Cabin", "Name", "Sex", "SibSp", "Parch", "Embarked", "Fare"], axis=1)

# assign features X and target value y
X = train_df.drop("Survived", axis=1).fillna(np.nan)
y = train_df.Survived
X_test = test_df.drop("PassengerId", axis=1).fillna(np.nan)


X.tail()


# In[18]:


# fill missing Age values through imputation
combined = pd.concat([X, X_test])
imp = Imputer()
imp_df = pd.DataFrame(imp.fit_transform(combined))
imp_df.columns = combined.columns
imp_df.index = combined.index
combined["Age"] = imp_df["Age"]
print(combined.isnull().sum())

X = combined[:len(X)]
X_test = combined[len(X):]
X.tail()


# In[19]:


# compare pipelines with different models through cross-validation
# logistic regression
log_pipeline = make_pipeline(LogisticRegression())
scores = cross_val_score(log_pipeline, X, y, scoring="neg_mean_absolute_error")
print('Logistic regression MAE: \t%2f' %(-1 * scores.mean()))

# random forest
rf_pipeline = make_pipeline(RandomForestRegressor())
scores = cross_val_score(rf_pipeline, X, y, scoring="neg_mean_absolute_error")
print('Random forest MAE: \t\t%2f' %(-1 * scores.mean()))

# XGBoost
xgb_pipeline = make_pipeline(XGBRegressor())
scores = cross_val_score(xgb_pipeline, X, y, scoring="neg_mean_absolute_error")
print('XGBoost MAE: \t\t\t%2f' %(-1 * scores.mean()))

# CatBoost
cat_pipeline = make_pipeline(CatBoostRegressor(verbose=False))
scores = cross_val_score(cat_pipeline, X, y, scoring="neg_mean_absolute_error")
print('CatBoost MAE: \t\t\t%2f' %(-1 * scores.mean()))

# naive Bayes
gnb_pipeline = make_pipeline(GaussianNB())
scores = cross_val_score(gnb_pipeline, X, y, scoring="neg_mean_absolute_error")
print('Naive Bayes MAE: \t\t%2f' %(-1 * scores.mean()))


# In[20]:


# use log. regression for submission (most successful)
log_pipeline.fit(X, y)
print("Accuracy: {}".format(cross_val_score(log_pipeline, X, y, scoring="accuracy").mean()))
y_pred = log_pipeline.predict(X_test)

result = pd.DataFrame()
result["PassengerId"] = test_df["PassengerId"].astype("int")
result["Survived"] = y_pred.astype("int")

result.to_csv("predicted_survival.csv", index=False)
result.head()

