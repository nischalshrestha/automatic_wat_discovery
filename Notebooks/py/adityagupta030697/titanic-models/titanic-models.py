#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import confusion_matrix, classification_report
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/titanic-vz-fe/train_new.csv")
test = pd.read_csv("../input/titanic-vz-fe/test_new.csv")
test_old = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


test_old.shape


# In[ ]:


# Splitting the data into Train/validation sets (25% val)
X = train.drop('Survived', axis = 1)
y = train.Survived
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(X,y,random_state=42)


# ## Logistic Regression
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

lr.fit(x_train,y_train)
print(lr.score(x_valid, y_valid),"\n")

lr_prediction = lr.predict(x_valid)
print(classification_report(y_valid, lr_prediction))
cm = pd.DataFrame(confusion_matrix(y_valid, lr_prediction), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm,"\n")


test_prediction = lr.predict(test)
PassengerId = test_old['PassengerId'].values
data = {'PassengerId' : PassengerId,'Survived' : test_prediction}
submission = pd.DataFrame(data)
submission.to_csv("Logreg_sub1.csv", index = False)
print(submission.head(),"\n")


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)
print(knn.score(x_valid, y_valid),"\n")

knn_prediction = knn.predict(x_valid)
print(classification_report(y_valid, knn_prediction))
cm = pd.DataFrame(confusion_matrix(y_valid, knn_prediction), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm,"\n")


# In[ ]:


from sklearn.model_selection import cross_val_score
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = [ x for x in myList if x%2!=0]

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    
MSE = [1 - x for x in cv_scores]    
import matplotlib.pyplot as plt
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)


# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# ## Final score after selecting number of neighbours
# 

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 29)
knn.fit(x_train,y_train)
print(knn.score(x_valid, y_valid),"\n")

knn_prediction = knn.predict(x_valid)
print(classification_report(y_valid, knn_prediction))
cm = pd.DataFrame(confusion_matrix(y_valid, knn_prediction), ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])
print(cm,"\n")


# In[ ]:


test_prediction = knn.predict(test)
data = {'PassengerId' : PassengerId,'Survived' : test_prediction}
submission = pd.DataFrame(data)
submission.to_csv("Knn_sub1.csv", index = False)
print(submission.head(),"\n")

