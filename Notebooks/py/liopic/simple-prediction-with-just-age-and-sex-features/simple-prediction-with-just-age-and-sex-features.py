#!/usr/bin/env python
# coding: utf-8

# This notebook shows in a simple way a quick process to train a ML model and submit the predictions it generates.
# 
# 
# 
# # 1. Loading the data

# In[7]:


# Import the usual libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
print(pd.__version__, np.__version__)


# We will load both train and test data (actually evaluation data), and concat them to work on both at the same time. Just notice that the test data has the _Survived_ feature missing.

# In[8]:


train_df = pd.read_csv('../input/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/test.csv', index_col='PassengerId')

df = pd.concat([train_df, test_df], sort=True)


# Let's see 10 random examples (if Survived is NaN, it's a one from the test/evaluation data)

# In[9]:


df.sample(10)


# You can refer to [its data dictionary](https://www.kaggle.com/c/titanic/data) to know more about these features.
# 
# Notice that original features start with uppercase. We will add later new features in lowercase.

# # 2. Exploration of age and sex features, and data preparation

# First let's see if the dataset has missing values.

# In[12]:


df[['Age', 'Sex']].isnull().sum()


# So we do need to fill in the missing Age values of 263 examples, and no need to do this with Sex feature.

# ## 2.1. Age
# Using pandas __.describe()__ method we can see general statistics for each feature.

# In[13]:


df['Age'].describe()


# In[27]:


# Quantity of people by given age
max_age = df['Age'].max()
df['Age'].hist(bins=int(max_age))


# In[29]:


# Survival ratio per decade, ignoring NaN with dropna()
df['decade'] = df['Age'].dropna().apply(lambda x: int(x/10))
df[['decade', 'Survived']].groupby('decade').mean().plot()


# The younger the passenger, the more chances of survival. There is some outsider at Age 80, however.
# 
# We need to complete missing values of Age. Let's do this using the mean value.

# In[31]:


mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)


# ## 2.2. Sex
# 
# Sex is stored as "male" or "female", but a ML algorithm needs to get numerical values as input. So let's create a new feature "male".

# In[32]:


df['male'] = df['Sex'].map({'male': 1, 'female': 0})
df.sample(5)


# In[35]:


df[['male','Survived']].groupby('male').mean()


# So 74% of females survived, while men had just a 18.9% of surviving ratio.

# # 3. Preparing the examples and training the ML algorithm
# 
# First we will prepare train examples for training the algorithm.

# In[43]:


train = df[df['Survived'].notnull()]

features = ['Age', 'male']
train_X = train[features]
train_y = train['Survived']


# Let's train a Decision Tree, which is really easy to understand.

# In[44]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(train_X, train_y)


# Let's display the tree.

# In[45]:


from sklearn.tree import export_graphviz
import graphviz
dot_data = export_graphviz(classifier, out_file=None,
                           feature_names=features,
                           class_names=['Dead', 'Alive'],  
                           filled=True, rounded=True)
graphviz.Source(dot_data)


# Basically it says: if the passenger is female, it will be alive; otherwise if the male is more than 6.5 years old, he will die. In other words "WOMEN and CHILDREN".
# 
# Let's see, with the same data used for training, the precision of this trained ML algorithm.

# In[42]:


classifier.score(train_X, train_y)


# So this trained algorith can predict 79% of correct results.

# # 4. Writting a submition with the test/evaluation data

# In[46]:


test = df[df['Survived'].isnull()]

test_X = test[features]
test_y = classifier.predict(test_X)


# In[47]:


submit = pd.DataFrame(test_y.astype(int),
                      index=test_X.index,
                      columns=['Survived'])
submit.head()


# Let's save this predictions in a file tha kaggle will use to evaluate it.

# In[48]:


submit.to_csv('submit.csv')


# This prediction will get a 75% of correct results.
# 
# # 5. Things to try
# 
# - Add more features
# - Improve the way the missing data is filled (for instance, use a separated mean Age for male and female)
# - Remove outsiders from the dataset
# - Consider using other ML classifier algorithms, like RandomForestClassifier
# - Adjust the hyperparameters of the ML algorithm
# - Consider ensembling several ML algorithms, with a voting system
