#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#load data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()


# # Data exploration, data cleaning and feature engineering

# In[ ]:


train.info()
print('***************************')
test.info()


# The, Age, Cabin and Embarked attributes have some missing values (less than 891 in trainin data and  less than 418 in test data). Because Cabin does not have impact in the prediction lets ignore it and focus on Age and Embarked.

# In[ ]:


# Sex (label encoding)
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)
test['Sex'].value_counts()
fig, axis1= plt.subplots(figsize=(8,3))
sns.countplot(x='Sex', data=train, ax=axis1)


# In[ ]:


#Embarked

embark = train['Embarked'].fillna('S')
train['Embarked'] = embark.map({'S': 1, 'C': 2, 'Q': 3}).astype(int)
test['Embarked'] = embark.map({'S': 1, 'C': 2, 'Q': 3}).astype(int)
sns.countplot(x='Embarked', data=train, ax=axis1)


# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(train['Age'].median())


# In[ ]:


X_train = train[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_train = train[['Survived']]


# In[ ]:


X_train.info()


# In[ ]:


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42)
sgd_clf.fit(X_train, y_train)


# In[ ]:


print(sgd_clf.predict(X_train[4:5]))
print(y_train[4:5])


# In[ ]:


cross_val_score(sgd_clf, X_train, y_train, cv = 10, scoring = 'accuracy')


# In[ ]:


y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)


# In[ ]:


confusion_matrix(y_train, y_train_pred)


# In[ ]:


precision_score(y_train, y_train_pred)


# In[ ]:


recall_score(y_train, y_train_pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, y_train)
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train, cv = 3)
print('precision_score',precision_score(y_train, y_train_pred_forest))
print('recall_score',recall_score(y_train, y_train_pred_forest))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)


# In[ ]:


y_train_pred_knn = cross_val_predict(knn_clf, X_train, y_train, cv = 3)
print('precision_score',precision_score(y_train, y_train_pred_knn))
print('recall_score',recall_score(y_train, y_train_pred_knn))


# In[ ]:


from sklearn.svm import SVC
svc_clf = SVC()
svc_clf.fit(X_train, y_train)


# In[ ]:


svc_clf_pred = cross_val_predict(svc_clf, X_train, y_train, cv = 3)
print('precision_score',precision_score(y_train, svc_clf_pred ))
print('recall_score',recall_score(y_train, svc_clf_pred ))


# In[ ]:


# only in test set fare feature has missing values
test['Fare'].fillna(test['Fare'].median(), inplace = True)


# In[ ]:


test['Age'] = test['Age'].astype(int)
test['Fare'] = test['Fare'].astype(int)


# In[ ]:


X_test = test[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]


# In[ ]:


y_test_pred = forest_clf.predict(X_test)

forest_clf.score(X_train, y_train)


# In[ ]:


forest_clf_scores = cross_val_score(forest_clf, X_train, y_train, cv = 10, scoring = 'accuracy')
forest_clf_scores.mean()


# In[ ]:




