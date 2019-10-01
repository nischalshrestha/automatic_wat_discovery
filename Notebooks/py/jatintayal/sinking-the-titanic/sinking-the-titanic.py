#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

#Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# **Step 1**: **Import the dataset.**
# 
# we first have to import the data we are going to be using. Dataset is diveded into two parts which are train, test.

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y_test = pd.read_csv("../input/gender_submission.csv")


# **Step 2:** **visualize the dataset**
# 
# Before we start writing the whole code we first have to know what our data is actually about because working on a machine learning algorithm without knowing what we are dealing with is the height of stupidity and we'll get to see some colourful graphs so it's a win-win ;)

# In[ ]:


train.count()


# In the output above we can see that there are 12 features and 891 training examples. But there is some data missing in 'Age', 'Cabin' which we'll clean later.

# In[ ]:


def visualize_data():    
    fig = plt.figure(figsize=(18, 6))
    
    plt.subplot2grid((2, 3), (0, 0))                #This plot shows the % of how many people died and how many survived.
    train.Survived.value_counts(normalize='true').plot(kind='bar')
    plt.title('Survivors')
    plt.xticks(np.arange(2), ('Deceased', "Survived"))

    plt.subplot2grid((2, 3), (0, 1), colspan=2)    #This plot shows the relation btw the age and class of the passanger.
    for x in [1, 2, 3]:
        train.Age[train.Pclass == x].plot(kind='kde')
    plt.legend(('1st', '2nd', '3rd'))
    plt.title('Age wrt class')

    plt.subplot2grid((2, 3), (1, 0))               #This plot shows the % of how many male passangers died and how many of them survived.
    train.Survived[train.Sex == "male"].value_counts(normalize='True').plot(kind='bar', color='b', alpha=0.5)
    plt.title('Male survivors')
    plt.xticks(np.arange(2), ('Deceased', 'Survived'))

    plt.subplot2grid((2, 3), (1, 1))               #This plot shows the % of how many female passangers died and how many of them survived.
    train.Survived[train.Sex == "female"].value_counts(normalize='True').plot(kind='bar', color='r', alpha=0.5)
    plt.title('Female survivors')
    plt.xticks(np.arange(2), ('Survived', 'Deceased'))

    plt.subplot2grid((2, 3), (1, 2))               #This plot shows the people survived on the basis of gender
    train.Sex[train.Survived == 1].value_counts(normalize='True').plot(kind='bar', color=['r', 'b'], alpha=0.5)
    plt.title('Sex of survivors')
    plt.xticks(np.arange(2), ('Female', "Male"))
    
    plt.show()
visualize_data()


# **Step 4: clean the dataset**
# 
# Our dataset has some holes in it as some values are missing and we have string values for some features. So in this function we'll clean our data and get rid of these anomalies.

# In[ ]:


train['Age'] = train['Age'].fillna(train['Age'].dropna().median()) #replace the NaN values in Age column with median.
test['Age'] = test['Age'].fillna(test['Age'].dropna().median())

train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)


# Now we are going' to convert our data into features(X) and target(Y) variables.

# In[ ]:


X_train = np.array(train[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'PassengerId']])        #Training set
Y_train = np.reshape(np.array(train['Survived']), (X_train.shape[0], 1))     #Target set

X_test = np.array(test[['Age', 'Sex', 'Pclass', 'SibSp', 'Parch', 'PassengerId']])   #Test set

print('Shape of X train:' + str(X_train.shape))
print('Shape of Y train:' + str(Y_train.shape))
print('Shape of X test:' + str(X_test.shape))
print('Shape of Y test:' + str(Y_test.shape))


# **Step 5: Train the model**
# 
# We are going to try some predefined scikit learn classifiers and then we'll choose which classifier works best for our data.

# **Model 1: KNeighborsClassifier**

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn_acc = round(knn.score(X_test, Y_test) * 100, 2)
print(knn_acc)


# **Model 2: Random forest classifier**

# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
rfc_acc = round(random_forest.score(X_test, Y_test) * 100, 2)
print(rfc_acc)


# **Model 3: Decision tree classifier**

# In[ ]:


decision_tree = tree.DecisionTreeClassifier(random_state=1, max_depth=7, min_samples_split=2)

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

dt_acc = round(decision_tree.score(X_test, Y_test) * 100, 2)
print(dt_acc)


# **Model 4: Support vector machine**

# In[ ]:


svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svm_acc = round(svc.score(X_test, Y_test) * 100, 2)
print(svm_acc)


# **Model evaluation**
# 
# We can now rank our evaluation of all the models to choose the best one for our problem. While both Decision Tree and Random Forest score the same, we choose to use Random Forest as they correct for decision trees' habit of overfitting to their training set.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Random Forest', 'Decision Tree'],
    'Score': [svm_acc, knn_acc, rfc_acc, dt_acc]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })


# In[ ]:




