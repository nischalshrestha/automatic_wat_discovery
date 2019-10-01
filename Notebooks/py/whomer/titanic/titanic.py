#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(train.shape, test.shape)
print(list(train.columns))
print(list(test.columns))


# In[ ]:


def plot_col_survived(col):
    pivot = train.pivot_table(index=col, values='Survived')
    pivot.plot.bar()
    plt.show()


# In[ ]:


plot_col_survived('Pclass')
plot_col_survived('Sex')
plot_col_survived('Age')
plot_col_survived('SibSp')
plot_col_survived('Parch')
plot_col_survived('Fare')
plot_col_survived('Embarked')


# ## feature engineering

# In[ ]:


train.head()


# In[ ]:


print(train.info())
print(test.info())


# Columns to use as features (still need further processing):  
# ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
# - as categories:
#     - Pclass
#     - Name: extract titles from names as categories
#     - Sex
#     - Age: as categories to handle null values
#     - Cabin: extract the first character as categories
#     - Embark
# - as numbesr: 
#     - SibSp
#     - Parch
#     - Fare: fill blank with mean

# In[ ]:


tarin_copy = train.copy()
test_copy = test.copy


# In[ ]:


title_map = {
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"
}
train['Title'] = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False).map(title_map)
test['Title'] = test["Name"].str.extract(' ([A-Za-z]+)\.',expand=False).map(title_map)
print(train.Title.value_counts())
print(test.Title.value_counts())


# In[ ]:


train['is_male'] = train.Sex.map({'male': 1, 'female':0})
test['is_male'] = test.Sex.map({'male': 1, 'female':0})


# In[ ]:


train.Age.fillna(-5, inplace=True)
test.Age.fillna(-5, inplace=True)


# In[ ]:


def cut_col(df, col, bins, labels):
    df['categorized_' + col] = pd.cut(df[col], bins=bins, labels=labels)
    return df


# In[ ]:


train[train.Survived == 0].Age.hist(alpha=.5, color='red', bins=20)
train[train.Survived == 1].Age.hist(alpha=.5, color='green', bins=20)
plt.legend(['Died', 'Survived'])


# In[ ]:


labels = ['Missing', '0-12', '12-18', '18-30', '30-55', '55-100']
bins = [-10, 0, 12, 18, 30, 55, 100]
train = cut_col(train, 'Age', bins, labels)
test = cut_col(test, 'Age', bins, labels)
train.columns


# In[ ]:


train.categorized_Age.value_counts()


# In[ ]:


train['clean_cabin'] = train.Cabin.str[0].fillna('Missing')
test['clean_cabin'] = test.Cabin.str[0].fillna('Missing')
most_common_embarked = train.Embarked.value_counts().index[0]
train['clean_embarked'] = train.Embarked.fillna(most_common_embarked)
test['clean_embarked'] = test.Embarked.fillna(most_common_embarked)


# In[ ]:


train.clean_embarked.value_counts()


# In[ ]:


def add_dummies(df, cols):
    dums = [pd.get_dummies(df[col], prefix=col) for col in cols]
    dfs = [df] + dums
    return pd.concat(dfs, axis=1)


# In[ ]:


columns_to_categories = ['Pclass', 'Title', 'categorized_Age', 'clean_cabin', 'clean_embarked']
train = add_dummies(train, columns_to_categories)
test = add_dummies(test, columns_to_categories)


# In[ ]:


test.columns


# In[ ]:


train.Fare.fillna(train.Fare.mean(), inplace=True)
test.Fare.fillna(test.Fare.mean(), inplace=True)


# In[ ]:


train.Fare.describe()


# In[ ]:


from sklearn.preprocessing import minmax_scale
cols_to_scale = ['SibSp', 'Parch', 'Fare']
for c in cols_to_scale:
    train[c + '_scaled'] = minmax_scale(train[[c]])
    test[c + '_scaled'] = minmax_scale(test[[c]])


# In[ ]:


features = ['SibSp_scaled', 'Parch_scaled', 'Fare_scaled', 'is_male','Pclass_1', 'Pclass_2', 'Title_Miss', 'Title_Mr',
       'Title_Mrs', 'Title_Officer', 'Title_Royalty', 'categorized_Age_0-12',
       'categorized_Age_12-18', 'categorized_Age_18-30',
       'categorized_Age_30-55', 'categorized_Age_55-100', 'clean_cabin_A',
       'clean_cabin_B', 'clean_cabin_C', 'clean_cabin_D', 'clean_cabin_E',
       'clean_cabin_F', 'clean_cabin_G',
       'clean_cabin_T', 'clean_embarked_Q',
       'clean_embarked_S']
target = 'Survived'
X = train[features]
y = train[target]


# In[ ]:


X.describe()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# # Make prediction using logistic regression:

# In[ ]:


lr = LogisticRegression(solver='liblinear')
scores = cross_val_score(lr, X, y, cv=10)
print(scores.mean())


# In[ ]:


X.shape


# In[ ]:


lr.fit(X, y)
for c in features:
    if c not in test.columns:
        test[c] = 0
predictions = lr.predict(test[features])


# In[ ]:


coef = lr.coef_
feature_importance = pd.Series(coef[0], index=features)
feature_importance.plot.barh()


# In[ ]:


from sklearn.feature_selection import RFECV
features = ['SibSp_scaled', 'Parch_scaled', 'Fare_scaled', 'is_male','Pclass_1', 'Pclass_2', 'Title_Miss', 'Title_Mr',
       'Title_Mrs', 'Title_Officer', 'Title_Royalty', 'categorized_Age_0-12',
       'categorized_Age_12-18', 'categorized_Age_18-30',
       'categorized_Age_30-55', 'categorized_Age_55-100', 'clean_cabin_A',
       'clean_cabin_B', 'clean_cabin_C', 'clean_cabin_D', 'clean_cabin_E',
       'clean_cabin_F', 'clean_cabin_G',
       'clean_cabin_T', 'clean_embarked_Q',
       'clean_embarked_S']
target = 'Survived'

X = train[features]
y = train[target]

lr = LogisticRegression()
selector = RFECV(lr, cv=10)
selector.fit(X, y)
optimized_features = X.columns[selector.support_]


# In[ ]:


print(optimized_features)


# In[ ]:


X = train[optimized_features]
y = train[target]

lr = LogisticRegression()
scores = cross_val_score(lr, X, y, cv=10)
print(scores.mean())


# In[ ]:


lr.fit(X, y)
for c in optimized_features:
    if c not in test.columns:
        test[c] = 0
predictions = lr.predict(test[optimized_features])


# In[ ]:


submission = pd.DataFrame({
    'PassengerId': test.PassengerId,
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)


# In[ ]:




