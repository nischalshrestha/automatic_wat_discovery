#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


# importing visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head()


# In[ ]:


sns.countplot('Pclass', data = train, hue = 'Sex')


# In[ ]:


sns.countplot('Survived', data = train, hue = 'Pclass')


# In[ ]:


sns.countplot('Survived', data = train, hue = 'Sex')


# In[ ]:


sns.distplot(train['Age'].dropna(), kde = False, hist_kws = dict(edgecolor = 'k', lw = 1), bins = 8)


# In[ ]:


# It seems that most of passengers were men at third class, Also the majority of non survival were men
# distribution of age skewed towards youth, Majority fall between 20 and 40 years old.
# majority of non-survival are from third class


# In[ ]:


# plotting a heatmap to identify null values
sns.heatmap(train.isnull(), yticklabels = False, cmap = 'YlGnBu')


# In[ ]:


# plotting box plot of Pclass agains ages to identify mean of ages
sns.boxplot(x = 'Pclass', y = 'Age', data = train)


# In[ ]:


# Average Age of Pclass 1 is 38, Pclass 2 is 29, Pclass 3 is 24
# replacing null age values by average ages of their Pclass
def avg_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age', 'Pclass']].apply(avg_age, axis = 1)


# In[ ]:


# dropping Cabin clumns 
train.drop('Cabin', axis = 1, inplace = True)


# In[ ]:


# plotting heat map of null values again
sns.heatmap(train.isnull(), yticklabels = False, cmap = 'YlGnBu')


# In[ ]:


# dropping any small null values as it'll not affect our model
train.dropna(inplace = True)


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


# Getting dummy variables of Sex, embark, SibSp, Parch
sex = pd.get_dummies(train['Sex'], drop_first = True)
embark = pd.get_dummies(train['Embarked'], drop_first = True)
sib_spo = pd.get_dummies(train['SibSp'], drop_first = True)
par_ch = pd.get_dummies(train['SibSp'], drop_first = True)
Pclass = pd.get_dummies(train['Pclass'], drop_first = True)


# In[ ]:


#dropping Name, Passenger Id, Ticket as they will not be included in our model and 
# replace new sex, par_ch, sib_spo and embark columns
train = train[['Age', 'Fare', 'Survived']]
train = pd.concat([train, sex, embark, sib_spo, par_ch, Pclass], axis = 1)


# In[ ]:


train.head()


# In[ ]:


# Dividing model to X and y
X = train.drop('Survived', axis = 1)
y = train['Survived']


# In[ ]:


# Since X values aren't on the same scale, Use standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)


# In[ ]:


scaled_X = scaler.transform(X)


# In[ ]:


X = pd.DataFrame(scaled_X)


# In[ ]:


X.head()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()


# In[ ]:


KNN.fit(X, y)


# In[ ]:


# applying same steps over titanic test data
test = pd.read_csv("../input/test.csv")


# In[ ]:


test['Age'] = test[['Age', 'Pclass']].apply(avg_age, axis = 1)
test.drop('Cabin', axis = 1, inplace = True)
test.dropna(inplace = True)
sex = pd.get_dummies(test['Sex'], drop_first = True)
embark = pd.get_dummies(test['Embarked'], drop_first = True)
sib_spo = pd.get_dummies(test['SibSp'], drop_first = True)
par_ch = pd.get_dummies(test['SibSp'], drop_first = True)
Pclass = pd.get_dummies(test['Pclass'], drop_first = True)


# In[ ]:


test = test[['Age', 'Fare']]
test = pd.concat([test, sex, embark, sib_spo, par_ch, Pclass], axis = 1)


# In[ ]:


scaler.fit(test)
scaled_test = scaler.transform(test)


# In[ ]:


test = pd.DataFrame(scaled_test)
test.head()


# In[ ]:


y_predict = KNN.predict(test)


# In[ ]:


y_predict


# In[ ]:




