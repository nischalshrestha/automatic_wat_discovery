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


train_data = pd.read_csv('../input/train.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.describe(include=['O'])


# In[ ]:


train_data.info()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


# In[ ]:


train_data.hist(bins=50, figsize=(20, 15))


# In[ ]:


def cate_survived(cate):
    return train_data[[cate, 'Survived']].groupby(cate, as_index=False).mean()


# In[ ]:


cate_survived('Pclass')


# In[ ]:


cate_survived('Sex')


# In[ ]:


cate_survived('SibSp')


# In[ ]:


cate_survived('Parch')


# In[ ]:


train_data.groupby(['Sex', 'Pclass']).Survived.mean().unstack()


# In[ ]:


train_data.groupby('Sex').Sex.describe()


# In[ ]:


train_data.pivot_table('Survived', 'Sex')


# In[ ]:


train_data.groupby(['Embarked']).Survived.describe()


# In[ ]:


g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[ ]:


corr_matrix = train_data.corr()
corr_matrix.Survived


# In[ ]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data.Survived


# In[ ]:


def preprocess_X(X):
    X = X.fillna(X.mean())
    X = pd.get_dummies(X)
    return X
X = preprocess_X(X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization


# In[ ]:


def nn_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(10,)))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# In[ ]:


model = nn_model()
model.fit(X_train, y_train, epochs=100)


# In[ ]:


print(model.metrics_names)
model.evaluate(X_test, y_test)


# In[ ]:


model = nn_model()
model.fit(X, y, epochs=100)


# In[ ]:


test_X = pd.read_csv('../input/test.csv')
test_X_final = preprocess_X(test_X[features])
test_y = model.predict_classes(test_X_final)
test_y = test_y.squeeze()
output = pd.DataFrame({
        'PassengerId': test_X.PassengerId,
        'Survived': test_y
    })
output.to_csv('submission_nn.csv', index=False)


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
scores = cross_val_score(model, X_final, y, cv=10)
print(scores.mean(), scores.std())


# In[ ]:


# Train, predict, write result
def formal_process():
    model = GradientBoostingClassifier()
    model.fit(X_final, y)
    test_data = pd.read_csv('../input/test.csv')
    test_X = preprocess_X(test_data[features])
    test_y = model.predict(test_X)
    output = pd.DataFrame({
        'PassengerId': test_data.PassengerId,
        'Survived': test_y
    })
    output.to_csv('submission.csv', index=False)
# formal_process()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


X_test.info()


# In[ ]:


from sklearn.impute import SimpleImputer
X_train_num = X_train.drop(['Sex', 'Embarked'], axis=1)
imputer = SimpleImputer()
X_train_num = imputer.fit_transform(X_train_num)
X_train_num = pd.DataFrame(X_train_num, columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'])
X_train_num


# In[ ]:


X_train_cat = X_train[['Sex', 'Embarked']].reset_index().drop('index', axis=1)
X_train_cat = pd.get_dummies(X_train_cat)
X_train_cat


# In[ ]:


X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
X_train_final


# In[ ]:


X_test_num = X_test.drop(['Sex', 'Embarked'], axis=1)
X_test_num = X_test_num.fillna(X_train_num.mean())
X_test_num


# In[ ]:


X_test_cat = X_test[['Sex', 'Embarked']]
X_test_cat = pd.get_dummies(X_test_cat)
X_test_cat


# In[ ]:


X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)
X_test_final


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_final, y_train)
model.score(X_test_final, y_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train_final, y_train)
model.score(X_test_final, y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train_final, y_train)
model.score(X_test_final, y_test)


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train_final, y_train)
model.score(X_test_final, y_test)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train_final, y_train)
model.score(X_test_final, y_test)


# In[ ]:


from sklearn import svm
model = svm.SVC()
model.fit(X_train_final, y_train)
model.score(X_test_final, y_test)


# In[ ]:


from sklearn import svm
model = svm.LinearSVC()
model.fit(X_train_final, y_train)
model.score(X_test_final, y_test)


# In[ ]:




