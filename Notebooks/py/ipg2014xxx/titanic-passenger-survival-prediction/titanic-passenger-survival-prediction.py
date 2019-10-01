#!/usr/bin/env python
# coding: utf-8

# ## **Getting Data**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib notebook')
sns.set(style="darkgrid")

# setting the random seed to get consistent results.
np.random.seed(7)


# In[2]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')

print("Training samples =", train.shape[0])
print("Testing samples =", test.shape[0])

train.head()


# ## Checking for null values
# 
# Now we check for the null values in both the training and test data.  So, we concatenate the training and testing samples. <br> 
# We see attributes like *Age, Cabin, Fare and Embarked* have null values.

# In[3]:


train_rows = train.shape[0]
y_train = train.pop('Survived')
test_id = test['PassengerId']

# merging the train and test set to apply preprocessing.
data = pd.concat([train, test])
del train
del test
data.isnull().sum()


# # Preprocessing and Visualization
# 
# * Dropping unnecessary columns like *PassengerId,* and *Ticket*
# * Filling up the null values as follows :
#     - in Age by median value.
#     - in Fare by mean value.
#     - in Embarked with most frequent value.
# * SibSp and Parch are combined to With_Family - 0 or 1
# * Normalizing the fare and the age.
# * Get dummies for the categorical attributes, * Sex, Embarked and Pclass*
# 
# 

# In[4]:


from sklearn.preprocessing import Imputer

data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis=1, inplace=True)

# filling the missing values in Age column with the median value
imputer = Imputer(strategy='median')
data['Age'] = imputer.fit_transform(data[['Age']])

# filling the missing values in Fare column with the mean value
imputer = Imputer(strategy='mean')
data['Fare'] = imputer.fit_transform(data[['Fare']])

# filling the missing values in Embarked column with most frequent value, i.e. 'S'
data.Embarked.fillna('S', inplace=True)

# Again checking for the null values.
data.isnull().sum()


# In[5]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
sns.distplot(data.Age, ax=ax[0])
sns.distplot(data.Fare, ax=ax[1], color='red')


# ### We see that most of the passengers are in the age group of 20-40.  Similarly, the fare seems to be around 0-100 for most passengers.

# In[6]:


#  Normalizing the Fare and Age values

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
data['Fare'] = np.log1p(data.Fare)
data['Age'] = mm.fit_transform(data[['Age']])


# In[7]:


data['With_Family'] = data.SibSp + data.Parch
data.With_Family = data.With_Family.apply(lambda x: 1 if x > 0 else 0)
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[8]:


train = pd.concat([data[:train_rows], y_train], axis=1)


# In[9]:


def annotate(ax):
    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('%d' %(y), (x.mean(), y), ha='center', va='bottom')
        
def plot_with_count(attribute):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    sns.countplot(x=attribute, data=train, ax=ax[0])
    annotate(ax[0])
    sns.countplot(x=attribute, data=train, ax=ax[1], hue='Sex')
    annotate(ax[1])
    sns.countplot(x=attribute, data=train, ax=ax[2], hue='Survived')
    annotate(ax[2])
    plt.tight_layout()


# In[10]:


plot_with_count(attribute='Pclass')


# In[11]:


plot_with_count(attribute='Embarked')


# In[12]:


plot_with_count(attribute='With_Family')


# In[13]:


data.head()


# In[14]:


data = pd.get_dummies(data=data, columns=['Pclass', 'Sex', 'Embarked'])
data.head()


# In[15]:


train = data[:train_rows]
test = data[train_rows:]

del data


# # Predicting the passenger survival

# In[16]:


from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

clf = XGBClassifier()
scores = cross_val_score(clf, train, y_train, cv=10)
np.mean(scores)


# In[17]:


clf.fit(train, y_train)


# In[18]:


'''import operator

feature_imp = {}
for feature, imp in zip(train.columns, clf.feature_importances_):
    feature_imp[feature] = imp
    
feature_imp = sorted(feature_imp.items(), key=operator.itemgetter(1), reverse=True)'''


# In[19]:


#feature_imp


# ## Predicting the results on test data

# In[20]:


y_pred = clf.predict(test)
y_pred = pd.Series(y_pred)
test_df = pd.DataFrame([test_id, y_pred]).transpose()
test_df.columns = ['PassengerId', 'Survived']
test_df.to_csv('submission.csv', index=False)

