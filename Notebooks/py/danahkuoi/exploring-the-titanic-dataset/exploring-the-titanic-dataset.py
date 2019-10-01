#!/usr/bin/env python
# coding: utf-8

# First draft notebook exploring the Titanic dataset using Python. Aim is to learn Python data analysis and visualisation techniques.

# In[ ]:


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


# In[ ]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
df = train
print (train.head())


# The above starts off our data exploration by displaying the header record of the training data set.

# In[ ]:


print (train.groupby('Pclass').mean())


# Using the above command, we can analyse the survival rate for each of the three Passenger Classes. So PClass '1' passengers had the;
# -	Highest survival rate, 62.96%
# -	The highest age, of 38 years old and;
# -	By far the most expensive fare, at 84pounds, 400% more than second class
# 
# It seems the Passenger Class you were in on the Titanic affected your odds or surviving. Unfortunately, there are reports that the lower classes were locked in the ship and not allow the chance to survive, which seems evident in the PClass survival stats.
# 
# Expanding on the PClass data, we can include the gender of survivals to determine if gender affected survival rate;

# In[ ]:


class_sex_grouping = (train.groupby(['Pclass','Sex']).mean())
print (class_sex_grouping)


# This clearly shows that females had a higher chance of surviving in all three passenger’s classes; 96.8% of females in first class survived, compared to 36.9% of males in the same class. The higher chances of female survivors is evident across all three Passenger Classes.
# 
# Using matplotlib, we can graph the above values to visually depict Passenger Class/Gender survival chances;

# In[ ]:


import matplotlib.pyplot as plt

class_sex_grouping['Survived'].plot.bar()
plt.show()


# Additionally, survival situations often follow the ‘women and children first’ mentality which can be seen when we use the following command to split survival by age;

# In[ ]:


group_by_age = pd.cut(train["Age"], np.arange(0, 90, 10))
age_grouping = train.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()
plt.show()


# The Age bars show survival rates in 10 year increments. Clearly the youngest age bracket, between 0-10 years of age, has a survival expectancy of just under 60%.

# Initial data analysis shows that the Passenger, Gender and Age are significant survival factors. Let's go back to the training data to determine all useful elements;

# In[ ]:


print (train.info())


# We can see that the training data that there are 891 passengers and data columns 'Age', 'Cabin', and 'Embarked' have less than 891 entries, therefore there is missing data. 
# 
# As there is so much missing 'Cabin' data, only 204 of 891 entries, we will drop this from the training data as I deem it not useful at the moment.
# 
# There are 714 'Age' entries out of 891, so it is missing 177 entries. We will fill these values by using the mean and/or median 'Age' values.
# 
# As there are only two missing 'Embarked' entries, we will fill the two missing entries with an value to represent 'missing'.

# In[ ]:


train = train.drop(['Name', 'Cabin', 'Ticket'], axis=1) 
train = train.dropna()
print (train.head())


# In[ ]:


print (train.info())


# Now we need to convert the 'Sex' and 'Embarked to integer values. We will convert Male to '0' and Female to '1'. The embarked values are;
# C = Cherbourg, which we will convert to '0'
# Q = Queenstown, which we will convert to '1'
# S = Southampton, which we will convert to '3'

# In[ ]:


#Create matrix for random forest classifier
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)
train.replace({'female':1,'male':0, 'S':1, 'C':2, 'Q':3}, inplace=True)
test.replace({'female':1,'male':0, 'S':1, 'C':2, 'Q':3}, inplace=True)

cols = ['Pclass','Age','SibSp', 'Embarked','Sex']
x_train = train[cols]
y_train = train['Survived']
#x_train, x_test, y_train, y_test = train_test_split(train[cols], train['Survived'], test_size=0.75, random_state=42)
x_test = test[cols]
id_test = test['PassengerId']

print("Training samples: {}".format(len(x_train)))
print("Testing samples: {}".format(len(y_train)))

#initialize the model
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
score = cross_val_score(model, x_train, y_train)
print("RandomForestClassifier :")
print(score)

output = pd.DataFrame(model.predict(x_test))
print(type(output))
print(type(id_test))
submission = pd.concat([id_test,output],axis=1)
submission.columns = ['PassengerId', 'Survived']

#Any files you save will be available in the output tab below
submission.to_csv('submission.csv', index=False)


# In[ ]:




