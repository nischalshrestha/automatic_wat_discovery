#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')



# In[2]:


##see the first 5 rows 
train_df.head()


# In[4]:


train_df.shape


# In[5]:


train_df.describe()


# In[6]:


#A heat map of correlation may give us a understanding of which variables are important
new_age = pd.DataFrame()
new_age['Age'] = train_df.Age.fillna(train_df.Age.mean())
new_age


# In[7]:


import seaborn as sns


# In[8]:


sns.countplot(x='Survived', data=train_df);


# In[10]:


sns.countplot(x='Sex', data=train_df);


# In[11]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=train_df)
#Females are more likely to survive than male


# In[12]:


train_df.groupby(['Sex']).Survived.sum()
#check how many males and females survived


# In[13]:


# Use pandas to figure out the proportion of women that survived, along with the proportion of men
print(train_df[train_df.Sex == 'female'].Survived.sum()/train_df[train_df.Sex == 'female'].Survived.count())
print(train_df[train_df.Sex == 'male'].Survived.sum()/train_df[train_df.Sex == 'male'].Survived.count())


# In[14]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Pclass'
sns.factorplot(x='Survived', col='Pclass', kind='count', data=train_df);
#Conclusion: 1st class passengers were most likely to survive 


# In[15]:


# Use seaborn to build bar plots of the Titanic dataset feature 'Survived' split (faceted) over the feature 'Embarked'
sns.factorplot(x='Survived', col='Embarked', kind='count', data=train_df);
## Passengers from Southampton were leess likely to survibve


# In[16]:


# Use a pandas plotting method to plot the column 'Fare' for each value of 'Survived' on the same plot.
train_df.groupby('Survived').Fare.hist(alpha=0.6);


# In[17]:


# Use the DataFrame method .describe() to check out summary statistics of 'Fare' as a function of survival
train_df.groupby('Survived').Fare.describe()


# In[18]:


# Use seaborn to plot a scatter plot of 'Age' against 'Fare', colored by 'Survived'
sns.lmplot(x='Age', y='Fare', hue='Survived', data=train_df, fit_reg=False, scatter_kws={'alpha':0.5});


# In[20]:


#It looks like those who survived either paid quite a bit for their ticket or they were young.
survived_train = train_df.Survived

# Concatenate training and test sets
data = pd.concat([train_df.drop(['Survived'], axis=1), test_df])

# Check out your new DataFrame data using the info() method
data.info()


# In[21]:


data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()


# In[22]:


# Encode the data with numbers because most machine learning models might require numerical inputs
# yo can do this using Pandas function get_dummies() which converts the categorical variable into numerical
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[23]:


# Select columns and view head
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[24]:


data.info()


# In[31]:


# Before fitting a model to your data, split it back into training and test sets
data_train = data.iloc[:891]
data_test = data.iloc[891:]
# A Scikit requirement transform the dataframes to arrays
X = data_train.values
test = data_test.values
y = survived_train.values


# In[32]:


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[33]:


# build your decision tree classifier with max_depth=3 and then fit it your data
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X, y)


# In[36]:


Y_pred = clf.predict(test)
test_df['Survived'] = Y_pred
test_df[['PassengerId', 'Survived']].to_csv('dec_tree.csv', index=False)


# In[ ]:




