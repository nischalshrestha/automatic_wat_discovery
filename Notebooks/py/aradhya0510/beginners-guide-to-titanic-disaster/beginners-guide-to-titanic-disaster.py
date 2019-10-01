#!/usr/bin/env python
# coding: utf-8

# This the simplest implementation of Scikit-Learn for Titanic Disaster (A gateway to Kaggle Competition). This kernel has been made with a sense that it should be as simple to understand as possible with minimum changes to dataset. I hope it helps.

# In[ ]:


import pandas as pd
import sklearn as sk


# In[ ]:


data_train = pd.read_csv('../input/train.csv') 


# In[ ]:


data_test = pd.read_csv('../input/test.csv')


# Let's have a look at the dataset first.

# In[ ]:


data_train.head()


# In[ ]:


data_train.head()


# There are are string and NaN values in data set which needs to be mapped and replaced 
# respectively. 

# Let's map all string values to corresponding numeric values so that classifier can understand them.

# In[ ]:


data_train['Sex'] = data_train['Sex'].map({'female': 1, 'male':0})
data_train['Embarked'] = data_train['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})


# Now we look for NaN values in each column.

# In[ ]:


data_train.isnull().sum()


# We need to fix the dataset with appropriate values in place of NaN by either deleting the column or replacing values. As in case of Age column we can find mean and replace all NaN values with it. (mean will be a decimal value so we take the nearest integer)

# In[ ]:


data_train.Age.mean()


# In[ ]:


data_train.Age.fillna(value=30,inplace=True)


# In[ ]:


data_train.Embarked.fillna(value=0,inplace=True)


# There are attributes in dataset that doesn't contribute anything to learners knowledge so can just delete them. (in a little  more complex approach these attributes can be transformed into something useful but we are trying to keep things simple here)

# In[ ]:


del data_train['Name']
del data_train['Ticket']
del data_train['Cabin']


# Now we repeat all the steps for test set.

# In[ ]:


data_test.head()


# In[ ]:


data_test.isnull().sum()


# In[ ]:


data_test['Sex'] = data_test['Sex'].map({'female': 1, 'male':0})
data_test['Embarked'] = data_test['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})


# In[ ]:


data_test.Age.mean()


# In[ ]:


data_test.Age.fillna(value=30,inplace=True)


# In[ ]:


data_test.Embarked.fillna(value=0,inplace=True)


# In[ ]:


data_test.Fare.mean()


# In[ ]:


data_test.Fare.fillna(value=35.6271884892086,inplace=True)


# In[ ]:


del data_test['Name']
del data_test['Ticket']
del data_test['Cabin']


# Let's see how data look now.

# In[ ]:


data_train.head()


# In[ ]:


data_test.head()


# Let's insert train and test in to classifier and get results.

# In[ ]:


train_cols = data_train.iloc[:,2:]
label_col = data_train.iloc[:,1]

X_train = train_cols
y_train = label_col

X_test = data_test.iloc[:,1:]


# In[ ]:


id_list = data_test.PassengerId


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)

train_features = X_train
train_target = y_train

clf = clf.fit(train_features, train_target)
pred = clf.predict(X_test)


# In[ ]:


Titanic_pred1 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred})


# In[ ]:


Titanic_pred1.head()


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model = model.fit(X_train,y_train)


pred2 = model.predict(X_test)


# In[ ]:


Titanic_pred2 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred})


# In[ ]:


Titanic_pred2.head()


# In[ ]:


from sklearn.ensemble import BaggingClassifier

clf = BaggingClassifier(n_estimators=1000)

train_features = X_train
train_target = y_train

clf = clf.fit(train_features, train_target)
pred3 = clf.predict(X_test)


# In[ ]:


Titanic_pred3 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred3})


# In[ ]:


Titanic_pred3.head()


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
pred4 = gnb.fit(X_train, y_train).predict(X_test)


# In[ ]:


Titanic_pred4 = pd.DataFrame({'PassengerId': id_list, 'Survived': pred4})


# In[ ]:


Titanic_pred4.head()


# We can see how different classifier are putting different results. 
# I hope this helps you in some way...
