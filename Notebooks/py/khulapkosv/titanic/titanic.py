#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessarry modules and libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# import scikit-learn libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.dropna()
df.info()


# In[ ]:


df.Age.describe()
#plt.hist(df['Age'])
#plt.show()


# In[ ]:


# Lets do first model
# Make a features 
feature_names = ['Pclass', 'SibSp', 'Parch']
X = df[feature_names]
X.info()


# In[ ]:


y = df['Survived']
y.shape


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RFC(random_state=0)
model.fit(X_train, y_train)


# In[ ]:


print(model.predict(X_test))


# In[ ]:


print("Training set accuracy: {:.3f}".format(model.score(X_train, y_train)))
print("Testing set accuracy: {:.3f}".format(model.score(X_test, y_test)))


# In[ ]:


# creating model for competition
test_df = pd.read_csv('../input/test.csv')
test_df.head()


# In[ ]:


feature_names = ['Pclass', 'SibSp', 'Parch']
test_X = test_df[feature_names]
test_X.dropna(axis=0)
test_X.info()


# In[ ]:


test_preds = model.predict(test_X)


# In[ ]:


output = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived': test_preds})
output.to_csv('submission.csv', index = False)

