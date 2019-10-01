#!/usr/bin/env python
# coding: utf-8

# This is my first classifier that I build for this competition

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Inputing the train and test data that will be used
titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')


# In[ ]:


#Data preprocessing before used in a neural network
titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())
titanic_train['Fare'] = titanic_train['Fare'].fillna(titanic_train['Fare'].median())
titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
titanic = titanic_train.append(titanic_test)
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
titanic_train = titanic[:891]
titanic_test = titanic[891:]
clf_train_output = titanic_train['Survived'].values.tolist()
clf_train_input = titanic_train[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
clf_test_input = titanic_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
#Feature Scaling
scaler = StandardScaler()
scaler.fit(clf_train_input)
clf_train_input = scaler.transform(clf_train_input)
clf_test_input = scaler.transform(clf_test_input)


# In[ ]:


#Training Classifier
clf = MLPClassifier(solver='lbfgs', alpha = 1e-4, hidden_layer_sizes = (6,3) , random_state = 1, max_iter = 5000)
clf.fit(clf_train_input,clf_train_output)


# In[ ]:


#Running Classifier onto the test dataset
clf_test_output = clf.predict(clf_test_input)
clf_test_output


# In[ ]:


#Comparing predicted to output to actual prediction
clf_accuracy = clf.score(clf_train_input,clf_train_output) * 100
clf_accuracy


# In[ ]:


#Submitting result
submission = pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": clf_test_output
})
submission.to_csv('submission.csv', index = False)
submission


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




