#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import xgboost as xgb 
from sklearn.model_selection import train_test_split
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# get titanic & test csv files as a DataFrame

#developmental data (train)
titanic_df = pd.read_csv("../input/train.csv")

#cross validation data (hold-out testing)
test_df    = pd.read_csv("../input/test.csv")

# preview developmental data cabin-船仓 embarked 上船 
titanic_df.head(5)
test_df.head(5)


# In[ ]:


#数据分布和缺失校验  描述每个字段的空
titanic_df.isnull().sum()
#求一下空值比例
round(177/(len(titanic_df["PassengerId"])),4)
# 补齐逻辑，可以是中位数，也可以试均值
ax = titanic_df["Age"].hist(bins=15, color='teal', alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()
titanic_df["Age"].median(skipna=True)


# In[ ]:


train_data = titanic_df
train_data["Age"].fillna(28, inplace=True)
train_data["Embarked"].fillna("S", inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)
train_data.drop('PassengerId', axis=1, inplace=True)
train_data.drop('Name', axis=1, inplace=True)
train_data.drop('Ticket', axis=1, inplace=True)


# In[ ]:


train_data.head(3)


# In[ ]:


Y=train_data['Survived']


# In[ ]:


one_hot_train_data=pd.get_dummies(train_data,columns=['Pclass','Sex','Embarked'])  #列枚举转编码
one_hot_train_data.drop('Survived', axis=1, inplace=True)

one_hot_train_data.head(3)


# In[ ]:


#make the test data on score
test_data = test_df
test_data["Age"].fillna(28, inplace=True)
test_data["Embarked"].fillna("S", inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)
test_data.drop('Ticket', axis=1, inplace=True)
test_X=pd.get_dummies(test_data,columns=['Pclass','Sex','Embarked'])  #列枚举转编码
test_X.drop('PassengerId', axis=1, inplace=True)


# In[ ]:


test_X.isnull().sum()
test_X["Fare"].fillna("10.0", inplace=True)


# In[ ]:


#SVM 分类
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)


# In[ ]:


clf.fit(X=one_hot_train_data,y=Y)


# In[ ]:


clf.predict(X=test_X)


# In[ ]:


# Gradient Boosting Classifier
from sklearn import ensemble
gradient_boost = ensemble.GradientBoostingClassifier(max_depth=6,n_estimators=10)
gradient_boost.fit(one_hot_train_data, Y)


# In[ ]:


gradient_boost.predict(X=test_X)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": clf.predict(X=test_X)
    })
submission.to_csv('titanic-svm.csv', index=False)
submission.head(10)


# In[ ]:


submission2 = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": gradient_boost.predict(X=test_X)
    })
submission2.to_csv('titanic-gradient_boost.csv', index=False)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(one_hot_train_data, Y, test_size=0.2, random_state=7)


# In[ ]:


model = xgb.XGBClassifier(max_depth=10, learning_rate=0.04, objective="binary:logistic",
                              silent=True,min_child_weight=6)
eval_data = [(X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_data, early_stopping_rounds=30)


# In[ ]:


X_train.head()


# In[ ]:


test_X.Fare=test_X.Fare.astype("float")


# In[ ]:


model.predict(test_X)


# In[ ]:


submission3 = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": model.predict(test_X)
    })
submission3.to_csv('titanic-xgboost.csv', index=False)


# In[ ]:




