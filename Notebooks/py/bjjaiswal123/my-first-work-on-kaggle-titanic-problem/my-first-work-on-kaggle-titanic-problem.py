#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame 
import matplotlib.pyplot as plt


# In[ ]:


import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# loading training and testing data as a DataFrame
'''
train = pd.read_csv("../kaggle/train.csv")
test = pd.read_csv("../kaggle/test.csv")
'''
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# viewing top 5 instances of training data
train.head()


# In[ ]:


test.info()


# In[ ]:


print(train.columns.values)


# In[ ]:


train.tail()


# In[ ]:





# In[ ]:


# we can see which columns are numerical and which are catogerical
train.info()
print('.'*40)
test.info()


# #### We can see that:
# * There are 5 object dtypes which means 5 categorical variables
# * They are:
#   * Name
#   * Sex
#   * Ticket
#   * Cabin
#   * Embarked

# ## Let's talk about basic steps Features Selection
# 
# ### It is necessary to understand features of dataset provided as it provides different benefits:
# * Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
# * Improves Accuracy: Less misleading data means modeling accuracy improves.
# * Reduces Training Time: Less data means that algorithms train faster.

# ## let's us understand our features and Survival relations

# ####  1. looking at the relationship between sex and survival.

# In[ ]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


sns.barplot(x="Sex", y="Survived", hue="Sex", data=train);


# #### Female surviving possibility is more than Male survival

# In[ ]:


sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train);


# ### We know from above that population from Embarked C has higher survival rate

# ### Plot survival rate by SibSp

# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train);


# ### Plot survival rate by Parch
# 

# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train);


# In[ ]:


train['Age'].hist(bins=70)


# ## Categorical features need to be transformed to numeric Features

# ### Why is it necessary to convert Categorical features to numeric Features?
# * There two basic reason for doing so:
#   1. This categorical Features also provide some useful information and helps in improving the prediction accuracy
#   2. Some of our model algorithms can only handle numeric values

# In[ ]:


train.info()
print('.'*40)
test.info()


# ### Let's us combine the whole training and testing for preprocessing so that we don't have to preprocess individually
# 

# In[ ]:



def preprocessing(data):
    data['Title']=pd.Series(data.Name.str.extract(' ([A-Za-z]+)\.', expand=False))
    for eachone in data:
        data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'other')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    data = data.drop(['Name','PassengerId','Cabin','Ticket'],axis =1)
    data['Title'] = data['Title'].replace({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "other": 5})
    data['Sex'] = data['Sex'].replace({"female":1,"male":0})
    
    for eachone in data:
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
        #filling Nan in age
        data['Age'] = data['Age'].fillna(data['Age'].mean())
        #filling Nan in Fare
        data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
        
    data['Embarked'] = data['Embarked'].replace({"S":1,"C":2,"Q":3})
    print(set(data['Embarked']))
    return data


# In[ ]:


train_new = preprocessing(train)
test_new = preprocessing(test)


# In[ ]:


train_new.info()
print('.'*40)
test_new.info()


# # let's get started to machine learning Model for Survival prediction

# ## Model, predict and solve
# 
# Now we are ready to train a model and predict the required solution. There are 60+ predictive modelling algorithms to choose from. We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...). We are also perfoming a category of machine learning which is called supervised learning as we are training our model with a given dataset. With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# 
#     Logistic Regression
#     KNN or k-Nearest Neighbors
#     Support Vector Machines
#     Naive Bayes classifier
#     Decision Tree
#     Random Forrest
#     Perceptron
#     Artificial neural network
# 
# 

# 

# In[ ]:


X_train = train_new.drop("Survived", axis=1)
Y_train = train_new["Survived"]
X_test  = test_new
X_train.shape, Y_train.shape, X_test.shape


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
acc_log = clf.score(X_train,Y_train)
print("Accuracy:",acc_log*100)


# # KNN or k-Nearest Neighbors
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
acc_knn= clf.score(X_train,Y_train)
print("Accuracy:",acc_knn*100)


# #  Support Vector Machines

# In[ ]:


from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# # Naive Bayes classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# # Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# # Model evaluation

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron,acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




