#!/usr/bin/env python
# coding: utf-8

# # Titanic 

# In[ ]:


# Imports
import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()

# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])

# View head
data.info()


# In[ ]:


data.Name.tail()


# In[ ]:


# Extract Title from Name, store in column and plot barplot
data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=data)
plt.xticks(rotation=45)


# In[ ]:


data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})
data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',
                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')
sns.countplot(x='Title', data=data);
plt.xticks(rotation=45);


# In[ ]:


data.tail()


# In[ ]:


# Did they have a Cabin?
data['Has_Cabin'] = ~data.Cabin.isnull()

# View head of data
data.head()


# In[ ]:


# Drop columns and view head
data.drop(['Cabin', 'Name', 'PassengerId', 'Ticket'], axis=1, inplace=True)
data.head()


# In[ ]:


data.info()


# In[ ]:


# Impute missing values for Age, Fare, Embarked
data['Age'] = data.Age.fillna(data.Age.mean())
data['Fare'] = data.Fare.fillna(data.Fare.mean())
data['Embarked'] = data['Embarked'].fillna('S')
data.info()


# In[ ]:


data.describe()


# In[ ]:


# Binning numerical columns
data['CatAge'] = pd.qcut(data.Age, q=6, labels=False )
data['CatFare']= pd.qcut(data.Fare, q=6, labels=False)
data.head()


# In[ ]:


data = data.drop(['Age', 'Fare'], axis=1)
data.head()


# In[ ]:


data['Fam_Size'] = data.Parch + data.SibSp
# Drop columns
data = data.drop(['SibSp','Parch'], axis=1)
data.head()


# In[ ]:


# Transform into binary variables
data_dum = pd.get_dummies(data, drop_first=True)
data_dum.head()


# In[ ]:


# Split into test.train
data_train = data_dum.iloc[:891]
data_test = data_dum.iloc[891:]

# Transform into arrays for scikit-learn
X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


from keras import models
from keras import layers
from keras import regularizers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(12, )))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X, y, epochs=150, batch_size=16)

Y_pred = model.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('titanic.csv', index=False)

