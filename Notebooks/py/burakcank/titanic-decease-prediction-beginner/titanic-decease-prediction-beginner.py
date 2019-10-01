#!/usr/bin/env python
# coding: utf-8

# **Titanic Decease Prediction (Beginner)**
# 
# *Burak Can Kahraman*
# 
# *02.08.2018*
# 
# Hey everyone, I am interested in the data science field and took a few courses online from MOOC websites. It seems like Kaggle is a great point to start and I just wanted to try my first submission, it may not be very accurate but still it's better than nothing :)
# 
# I will improve this notebook over time to have a more accurate result, but for now bare with me.

# In[ ]:


# import the libraries
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# import the dataset
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

df.head(10)


# In[ ]:


# feature selection - exclude passenger name, ticket and cabin for now
df = df.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9, 11]]
test = test.iloc[:, [0, 1, 3, 4, 5, 6, 8, 10]]

df.head(10)


# In[ ]:


# inspect the data
df.describe()


# In[ ]:


# look which values are missing in the training set
df.info()


# In[ ]:


# split the data into features and labels
X_train = df.drop(columns=['Survived', 'PassengerId'])
y_train = df['Survived']

# take age as numpy array of shape (?, 1)
age = X_train['Age'].values.reshape(-1, 1)


# In[ ]:


# fill in the missing values
imp = SimpleImputer()
X_train['Age'] = imp.fit_transform(age)

X_train['Embarked'] = X_train['Embarked'].fillna(method = 'ffill')


# In[ ]:


# look which values are missing in the test set
test.info()
test.head()


# In[ ]:


# fill in the missing values in test set
test = test.drop(columns=['PassengerId'])

test_age = test['Age'].values.reshape(-1, 1)
test['Age'] = imp.fit_transform(test_age)

test_fare = test['Fare'].values.reshape(-1, 1)
test['Fare'] = imp.fit_transform(test_fare)

test.head(10)


# In[ ]:


# label binarize column 'Embarked' and 'Sex' in training data
lb = LabelBinarizer()
lb_embarked = lb.fit_transform(X_train['Embarked'])
lb_embarked = pd.DataFrame(lb_embarked, columns=lb.classes_)
X_train = X_train.drop(columns=['Embarked']).join(lb_embarked)

X_train['Sex'] = lb.fit_transform(X_train['Sex'])
X_train.head(10)


# In[ ]:


# label binarize test data
lb_embarked = lb.fit_transform(test['Embarked'])
lb_embarked = pd.DataFrame(lb_embarked, columns=lb.classes_)
test = test.drop(columns=['Embarked']).join(lb_embarked)

test['Sex'] = lb.fit_transform(test['Sex'])
test.head(10)


# In[ ]:


# scale the train data
scaler = MinMaxScaler()
X_train['Age'] = scaler.fit_transform(X_train['Age'].values.reshape(-1, 1))
X_train['Fare'] = scaler.fit_transform(X_train['Fare'].values.reshape(-1, 1))
X_train.head()


# In[ ]:


# scale the test data
test['Age'] = scaler.fit_transform(test['Age'].values.reshape(-1, 1))
test['Fare'] = scaler.fit_transform(test['Fare'].values.reshape(-1, 1))
test.head()


# In[ ]:


# # train the model with Multiple Linear Regression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# # predict the results
# y_pred = regressor.predict(test)

# # apply logistic regression
# result = [1 if item > 0.5 else 0 for item in y_pred]

# index = np.arange(892, 1310)
# b = np.column_stack((index, result))

# result_df = pd.DataFrame(b, columns = ['PassengerId', 'Survived'])

# result_df.head(10)
# result_df.to_csv('result.csv', index=False)


# In[ ]:


# train the model with Stochastic Gradient Descent
classifier = SGDClassifier()
classifier.fit(X_train, y_train)

# predict the results
y_pred = classifier.predict(test)

index = np.arange(892, 1310)
result = np.column_stack((index, y_pred))

result_df = pd.DataFrame(result, columns = ['PassengerId', 'Survived'])

result_df.head(10)
result_df.to_csv('resultSGD.csv', index=False)


# This prediction has %75 correctness, I guess this should be better than nothing, at least compared to random prediction. But again this is just the first submission, it is going to be updated, until then :)
