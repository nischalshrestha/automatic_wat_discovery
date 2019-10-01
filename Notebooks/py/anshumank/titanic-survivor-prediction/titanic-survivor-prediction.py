#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivor Prediction

# Importing all Libraries

# In[5]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import array
from sklearn.metrics import accuracy_score
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix


# Input Training Values

# In[ ]:


train = pd.read_csv ("../input/titanic/train.csv")


# In[ ]:


train.head()


# In[ ]:


train.info()


# Cabin, Age, Embarked have missing values.
# Cabin has more than 20% of data to be missing so we just drop the coulmn.
# For Age and Embarked we remove corresponding column with missing values.

# In[ ]:


del train['Cabin']


# In[ ]:


train=train.dropna(subset = ['Age', 'Embarked'])


# In[ ]:


train.info()


# All the coulmn have equal number of non-null values

# In[ ]:


train = train.drop (columns=['Name'])


# In[ ]:


train = train.drop (columns=['PassengerId'])


# In[ ]:


train = train.drop (columns=['Ticket'])


# In[ ]:


train.info()


# ## Basic Visualizations using (Tableau)

# ### This clearly shows that the lesser people survived the tragic accident. 

# In[1]:


from IPython.display import Image
Image(filename='../input/titanic-visualization/Count of People Survived.png')


# ### The priority to rescue women can be seen. More fraction of women survived the accident

# In[2]:


Image(filename='../input/titanic-visualization/Count of People Survived based on Sex.png')


# ### The graph below shows the count of people with respect to their embarkment station. C = Cherbourg, Q = Queenstown, S = Southampton

# In[3]:


Image(filename='../input/titanic-visualization/Count of People Survived based on Embarked.png')


# ## Continuing to implement algorithm

# It is important to note that the Sex and Embarked column are object and have sting values.
# Therefore they cannot be directly be replaced by the numbers. (May cause bias in the algorithm)
# To counter the problem we use **One Hot Encoder** using *get_dummies()*

# In[ ]:


train_dummies = pd.get_dummies (train)
train_dummies.info()


# In[ ]:


X = train_dummies.iloc[:, [1,2,3,4,5,6,7,8,9,10]].values
Y = train_dummies.iloc[:, [0]].values


# In[ ]:


X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.1, random_state = 0)


# Normalizing the values using StandardScaler()

# In[ ]:


sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)


# In[ ]:


#Classification Using Random Forest
classifier = RandomForestClassifier(n_estimators = 80, criterion = 'entropy', max_depth = 6)
classifier.fit(X_Train,Y_Train.ravel())


# In[ ]:


Y_Pred = classifier.predict(X_Test)


# In[ ]:


Y_Pred


# In[ ]:


cm = confusion_matrix(Y_Test, Y_Pred)


# In[ ]:


cm


# The test set has been changed: all the missing values in the Age, Fare have columns have been inserted with mean value. 

# In[ ]:


test = pd.read_csv ("../input/titanic-testing-set-without-missing-values/test.csv")
test = test.drop (columns=['Name'])
test = test.drop (columns=['PassengerId'])
test = test.drop (columns=['Ticket'])
test = test.drop (columns=['Cabin'])


# In[ ]:


test.info()


# In[ ]:


test_dummies = pd.get_dummies (test)
test_dummies.info()


# In[ ]:


X = test_dummies.iloc[:, [0,1,2,3,4,5,6,7,8,9]].values
test1 = pd.read_csv ("../input/titanic/gender_submission.csv")
Y = test1.iloc[:, [1]].values


# In[ ]:


sc_X = StandardScaler()
X_Test = sc_X.fit_transform(X)


# In[ ]:


Y_Pred = classifier.predict(X_Test)


# In[ ]:


cm = confusion_matrix(Y, Y_Pred)


# In[ ]:


cm


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




