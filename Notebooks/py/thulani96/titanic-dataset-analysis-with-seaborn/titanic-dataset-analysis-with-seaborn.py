#!/usr/bin/env python
# coding: utf-8

# ### Step I: Import Libraries for Data Visualization

# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import rcParams
get_ipython().magic(u'matplotlib inline')
# figure size in inches
rcParams['figure.figsize'] = 15,6


# ### Step II: Load The Titanic DataSet

# In[ ]:


data = pd.read_csv('../input/train.csv')
data.fillna(0, inplace = True)
# Convert the survived column to strings for easier reading
data['Survived'] = data['Survived'].map({
    0: 'Died',
    1: 'Survived'
})

# Convert the Embarked column to strings for easier reading
data['Embarked'] = data['Embarked'].map({
    'C':'Cherbourg',
    'Q':'Queenstown',
    'S':'Southampton',
})

data.head()


# ### Step III: Start The Visualization
# First We Look at how many passengers Lived or Dies in each Passenger Class

# In[ ]:


# fig, ax = plt.subplots(1,1, figsize = (12,10))
ax = sns.countplot(x = 'Pclass', hue = 'Survived', palette = 'Set1', data = data)
ax.set(title = 'Passenger status (Survived/Died) against Passenger Class', 
       xlabel = 'Passenger Class', ylabel = 'Total')
plt.show()


# #### Visualization II: We look how many people lived or Died according to their Sex

# In[ ]:


print(pd.crosstab(data["Sex"],data.Survived))
ax = sns.countplot(x = 'Sex', hue = 'Survived', palette = 'Set1', data = data)
ax.set(title = 'Total Survivors According to Sex', xlabel = 'Sex', ylabel='Total')
plt.show()


# #### Visualization III: Next we look at Survivors with regards to Age groups

# In[ ]:


# We look at Age column and set Intevals on the ages and the map them to their categories as
# (Children, Teen, Adult, Old)
interval = (0,18,35,60,120)
categories = ['Children','Teens','Adult', 'Old']
data['Age_cats'] = pd.cut(data.Age, interval, labels = categories)

ax = sns.countplot(x = 'Age_cats',  data = data, hue = 'Survived', palette = 'Set1')

ax.set(xlabel='Age Categorical', ylabel='Total',
       title="Age Categorical Survival Distribution")

plt.show()


# #### Visualiztion IV: We get get to see the survival distribution based on where passengers embarked from

# In[ ]:


print(pd.crosstab(data['Embarked'], data.Survived))
ax = sns.countplot(x = 'Embarked', hue = 'Survived', palette = 'Set1', data = data)
ax.set(title = 'Survival distribution according to Embarking place')
plt.show()


# In[ ]:


# print(data.nunique())
data.head()


# ### Proprocessing: Drop Unwanted Features

# In[ ]:


data.drop(['Name','Ticket','Cabin','PassengerId','Age_cats'], 1, inplace =True)
# data.fillna(0, inplace = True)
data.head()


# In[ ]:


data.Sex.replace(('male','female'), (0,1), inplace = True)
data.Embarked.replace(('Southampton','Cherbourg','Queenstown'), (0,1,2), inplace = True)
data.Survived.replace(('Died','Survived'), (0,1), inplace = True)
data.head()


# ### Lets see the Correlation between the features

# In[ ]:


plt.figure(figsize=(14,12))
sns.heatmap(data.astype(float).corr(),linewidths=0.1, 
            square=True,  linecolor='white', annot=True)
plt.show()


# In[ ]:



data.head()


# ### Prepare X (Features) & y (Labels) for the classifier

# In[ ]:


X = np.array(data.drop(['Survived'],1))
y = np.array(data['Survived'])
print("Features shape: ", X.shape)
print("Labels: ", y.shape)


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
data.fillna(0, inplace = True)
sns.heatmap(data.isnull())
data.head()


# ### Import Machine Learning Libraries

# In[ ]:


from sklearn.cluster import KMeans
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score


# In[ ]:





# ### P.S, this is my first Notebook and Exercise with Machine Learning and Data analysis, i'm still working on the notebook. Do pass your suggestions on wherever i can make improvements in the meantime
