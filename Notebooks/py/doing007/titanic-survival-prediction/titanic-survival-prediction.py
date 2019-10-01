#!/usr/bin/env python
# coding: utf-8

# This is my first Notebook in Kaggle, This code is inspired by the https://www.kaggle.com/sinakhorami/titanic-best-working-classifier notebook

# ## Import necessary libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

get_ipython().magic(u'matplotlib inline')


# ## Importing train data test pandas DataFrame

# In[ ]:


os.listdir('../')

# Importing train dataset
train = pd.read_csv('../input/train.csv')

# To see the no of rows and columns in the dataframe
print(train.shape)

# Describe dataframe to see more information about the data and missing values
train.describe() # train_df.info() also can be used

#  We can notice DF total have 891 rows and Age is missing in some rows


# In[ ]:


# Importing testing dataset
test = pd.read_csv('../input/test.csv')

print(test.shape)
print(test.info())


# In[ ]:


# Printig some rows from train
train.head()


# ## Data Visualization to understand the data

# In[ ]:


sex_plt = train.pivot_table(index='Sex', values='Survived')
sex_plt.plot.bar()


# In[ ]:


sex_plt = train.pivot_table(index='Pclass', values='Survived')
sex_plt.plot.bar()
# As you can see, If passenger is women or passenger belongs to 1st class has better chance of surviving


# In[ ]:





# ### Checking If DataFrame has any missing Data

# In[ ]:


# Data missing for Age in the train dataset
print(train.shape)
display(train.describe())

# Using pandas.fillna method to fill the NaN values using Age mean
train['Age'].fillna(train.Age.mean(), inplace=True)


# ### Checking If Data frame has any missing Data and filling

# In[ ]:



print(test.shape)
display(test.describe())

# Using pandas.fillna method to fill the NaN values using Fare median
test['Fare'].fillna(test.Fare.median(), inplace=True)
test['Age'].fillna(test.Age.mean(), inplace=True)


# ### Removing the features which are not impacting more on the Survival decission

# In[ ]:


# For Train data
drop_features = ['Name', 'PassengerId', 'Ticket', 'Cabin']
train = train.drop(drop_features, axis = 1)
train = pd.get_dummies(train, drop_first=True)

display(train.head())


# In[ ]:


# For Test data
test = test.drop(drop_features, axis = 1)
test = pd.get_dummies(test, drop_first=True)
display(test.head())


# ### Creating OneHotEncoder using pandas get_dummies - since machine learning handles only with numbers

# In[ ]:





# In[ ]:


train.describe()


# ### Now we have Final train data is ready, Lets start train out models

# In[ ]:


train.head()


# ## Lets use multiple algorithms for training and select the best out of it

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss


# ### Trying different Classifiers to see which one gives good accuracy based on training dataset

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()
]


# In[ ]:


log_cols = ['Classifier', 'Accuracy']
log = pd.DataFrame(columns=log_cols)


# In[ ]:


sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

X = train.iloc[:, 1:].values
y = train.iloc[:, 0].values

print("X ", X.shape)
print("y ", y.shape)

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        
        y_prediction = clf.predict(X_test)
        acc = accuracy_score(y_test, y_prediction)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc


# In[ ]:


for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)
    


# In[ ]:


plt.xlabel = "Accuracy"
plt.title = "Clasifier Accurcy"

sns.set_color_codes("muted")


# In[ ]:


sns.barplot(x="Accuracy", y="Classifier", data=log, color="b")


# ### GradientBoostingClassifier is the winner here
# I noticed that *GradientBoostingClassifier* is working better with this data and we use this Classifier to predict

# In[ ]:


output_cols = ['PassengerId', 'Survived']

high_acc_clf = GradientBoostingClassifier()
high_acc_clf.fit(X, y)

y_pred = high_acc_clf.predict(test.values)
y_pred

test_data = pd.read_csv('../input/test.csv')
data = {
    "PassengerId": test_data.PassengerId,
    "Survived": y_pred
}
data
new_df = pd.DataFrame(data)
new_df.to_csv('./prediction.csv', index=False)


# In[ ]:





# In[ ]:




