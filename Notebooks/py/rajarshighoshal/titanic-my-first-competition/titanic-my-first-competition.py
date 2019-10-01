#!/usr/bin/env python
# coding: utf-8

# ## Importing basic libraries

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


# ## Loading the data

# In[ ]:


df_train = pd.read_csv('../input/train.csv') # trainning dataset
df_test = pd.read_csv('../input/test.csv') # testing data set


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train.describe()


# In[ ]:


# getting total no. of passangers in training data
print('Total number of passengers in training data: ', len(df_train))
# getting no. of passangers who have survived
print('Total number of passengers survived: ', len(df_train[df_train['Survived'] == 1]))


# In[ ]:


# getting % of male and female survived to watch if sex played some role
print('% of male survived: ', 100*np.mean(df_train['Survived'][df_train['Sex'] == 'male']))
print('% of female survived: ', 100*np.mean(df_train['Survived'][df_train['Sex'] == 'female']))


# We can see 74% of female is survived while only approx. 19% of male is survived. So, sex played a role here.

# In[ ]:


# getting % of people who servived above anb below age 15
print('% of children who survived: ', 100*np.mean(df_train['Survived'][df_train['Age'] < 15]))
print('% of adults who survived: ', 100*np.mean(df_train['Survived'][df_train['Age'] > 15]))


# So, age played a role as well.

# In[ ]:


# comparing survival rate of different class people
print('% of 1st class passengers who survivde: ', 100*np.mean(df_train['Survived'][df_train['Pclass'] == 1]))
print('% of 2nd class passengers who survivde: ', 100*np.mean(df_train['Survived'][df_train['Pclass'] == 2]))
print('% of 3rd class passengers who survivde: ', 100*np.mean(df_train['Survived'][df_train['Pclass'] == 3]))


# So, we cann see that class of passanger is a factor as well.

# In[ ]:


# what abot SibSp and Parch
print('% of passengers with parents who survivde: ', 100*np.mean(df_train['Survived'][df_train['Parch'] >= 1]))
print('% of passengers without parents who survivde: ', 100*np.mean(df_train['Survived'][df_train['Parch'] < 1]))
print('% of passengers with siblings who survivde: ', 100*np.mean(df_train['Survived'][df_train['SibSp'] >= 1]))
print('% of passengers without siblings who survivde: ', 100*np.mean(df_train['Survived'][df_train['Parch'] < 1]))


# We can consider SibSp and Parch as factor as well

# ## Processing the data for analysis

# Dealing with non-numerical sex and converting it to binary values

# In[ ]:


# female = 1, male = 0
df_train['Sex'] = df_train['Sex'].apply(lambda x: 1 if x == 'female' else 0)
df_train.head()


# In[ ]:


# checking nan vlaues
df_train.isnull()
# handling nan values for age and fare
df_train['Age'] = df_train["Age"].fillna(np.mean(df_train['Age']))
df_train['Fare'] = df_train['Fare'].fillna(np.mean(df_train['Fare']))


# In[ ]:


# showing all the columns
df_train.columns


# In[ ]:


# columns which are useful for analysis
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']


# In[ ]:


# get training df
train = df_train[features]
train.head()


# ## spliting factors and outcoms
# Get X and yas factors and outcome dataframe

# In[ ]:


X = train
y = df_train['Survived']


# Now spliting data to train and test 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# ## Training the model
# train using decisiontreeregressor model

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 1)
model.fit(X_train, y_train)


# ## Testing MAE
# 

# In[ ]:


from sklearn.metrics import mean_absolute_error
print('for training set: ', mean_absolute_error(y_train, model.predict(X_train)))
print('for testing set: ', mean_absolute_error(y_test, model.predict(X_test)))


# ## Submission
# 

# In[ ]:


test = df_test.copy()[features]
test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'female' else 0)
test['Age'] = test["Age"].fillna(np.mean(test['Age']))
test['Fare'] = test['Fare'].fillna(np.mean(test['Fare']))
model.fit(X,y)
predict_survival = model.predict(test)
predict_survival = np.round(predict_survival)
predict_survival = predict_survival.astype(int)


# In[ ]:


submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predict_survival})

submission.to_csv('submission.csv', index=False)

