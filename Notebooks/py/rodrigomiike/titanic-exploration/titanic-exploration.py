#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# ## Introduction
# Titanic is a famous ship that sank on 15 April 1912 after colliding with an iceberg. 
# There were 2224 passengers and 1502 died making it one of the deadliest disaster of the modern history. <sup>[1]</sup>
# 
# Our job here is to build a model that answers how likely people were to survive this disaster. <br />
# The process we'll be as following:
# 1. Data Exploration and Visualization
# 2. Feature Engineering
# 3. Making predictions
# 4. Tuning hyperparameters
# 5. Conclusion
# 
# ## Source
# 1. https://en.wikipedia.org/wiki/RMS_Titanic

# In[2]:


# for text patterns
import re
# for math stuff
import numpy as np
# for handling the dataset
import pandas as pd
# for data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# model used for classification
from sklearn.ensemble import RandomForestClassifier
# metric used to measure the performance of the classifier
from sklearn.metrics import accuracy_score

# for reproducibility
np.random.seed(0)

sns.set(style="white", context="talk")
get_ipython().magic(u'matplotlib inline')


# ## Data Exploration and Visualization
# First of all, let's load and explore the dataset.

# In[3]:


train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

print('train size: {0}; test size: {1}'.format(len(train), len(test)))


# We have a training set with 891 samples and a testing set with 418 samples. <br />
# Let's take a preview on training set.

# In[4]:


train.head()


# Here, we can see that our response variable (label) is **Survived** and it's a categorical variable. <br />
# This means that we'll be using a classifier as prediction model.
# 
# The other ones are explanatory variables and there are categorical and continuous variables.
# 
# We can see the feature **Cabin** has NaN values. This is a problem the should be handled. <br />
# Let's take an overview on dataset and see the quantity of NaN values.

# In[5]:


print('***TRAINING SET***')
print(train.isnull().sum())
print('\n***TESTING SET***')
print(test.isnull().sum())


# Now we can see that:
# * The feature **Cabin** might be discarded since more than 70% of the data isn't provided.
# * The feature **Embarked** has only 2 samples with NaN. We could discard the sample or impute values.
# * The feature **Age** has 263 samples with NaN. This is about 20% of the data. Discarding the feature might be pretty much loss of information. So, we should be imputing new values.
# 
# Now, let's plot some graphs in order to get some insight about the dataset. <br />
# Our response variable is **Survived**, so let's see the ratio between the Survived and Desceased people.

# In[6]:


plt.title('Survived x Deceased')
sns.countplot(data=train, x='Survived')


# Here we can see that most of the people desceased, but nothing more. <br />
# Let's see the survivability rate between genders.

# In[7]:


sns.factorplot(data=train, x='Survived', hue='Sex', kind='count')
plt.title('Survivability between genders')


# Now, we have an interesting insight. Most of the men died on this disaster. <br />
# This is due to the "women and children first" protocol while loading the safeboats.

# In[8]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

sns.countplot(data=train, x='Pclass', hue='Sex', ax=ax1)
ax1.set_title('Survivability between Classes|Gender')

sns.countplot(data=train, x='Pclass', hue='Survived', ax=ax2)
ax2.set_title('Survivability between Classes')


# In[9]:


ax = sns.factorplot(data=train, x='Survived', hue='Sex', col='Pclass', kind='count')
ax.fig.subplots_adjust(top=.8)
ax.fig.suptitle('Survivability between Gender|Class')


# After we factored the plot between classes, it seems that first class were more likely to survive, even though the third class had more passengers. <br />

# In[10]:


# train[(train.SibSp > 0)|(train.Parch > 0)]
# train[((train.SibSp > 0)|(train.Parch > 0))&(train.Pclass == 1)]
train[((train.Parch > 1))&(train.Pclass == 1)]


# In[11]:


train[train.Ticket=='CA. 2343']


# An interesting fact is that families generally bought the same ticket. <br />

# In[12]:


# train[train.Ticket=='2666']
# train[train.Ticket=='19950']
train[train.Ticket=='113760']


# And after taking a look on some families, there are cases that:
# * The whole family sank together;
# * The whole family survived together;
# * Only the females of the family survived;

# ---
# ## Feature Engineering
# ### Dealing with missing values
# Before starting the feature engineering, we'll be merging both datasets in one.

# In[13]:


X, y = train.iloc[:,2:], train.iloc[:,1] # separating the labels


# In[14]:


X = pd.concat([X[:], test.iloc[:,1:][:]], ignore_index=True) # merging the datasets


# In[15]:


print('total is {}'.format(len(X)))


# Let's start with the easiest one. <br />
# Embarked have only 2 samples with NaN, let's take a look on them.

# In[16]:


X[X.Embarked.isnull()]


# We can see that both of them are somehow related. <br />
# They have the same ticket number, paid the same fare and shared the same cabin. Consequently, they might Embarked from the same place. <br />
# We'll running a *Random Forest* here to impute **Embarked** since the classifier requires almost no feature engineering. <br />

# In[17]:


_ = X[~X.Embarked.isnull() & ~X.Fare.isnull()][['Pclass', 'Fare', 'Embarked']].as_matrix()
clf = RandomForestClassifier()
clf.fit(_[:,:2], _[:,2])


# In[18]:


clf.predict(X[X.Embarked.isnull()][['Pclass', 'Fare']].as_matrix())


# In[19]:


X.loc[[61, 829],['Embarked']] = 'S'


# In[20]:


X.loc[[61, 829]]


# The classifier predicted as they embarked from Southampton, so we'll be imputing 'S' to them.
# 
# The feature **Cabin** should be discarded and I'll be doing some feature engineering on the feature **Name**. <br />
# So, let's start by extracting the title from each of them.

# In[21]:


del X['Cabin']
# del X['Name']


# In[22]:


title_ptr = re.compile('\w+?\.')
def get_title(s):
    m = title_ptr.search(s)
    return m.group()


# In[23]:


X['titles'] = X.Name.apply(get_title)


# In[24]:


plt.figure(figsize=(15, 8))
plt.title('Titles')
sns.countplot(data=X, x='titles')


# We can see that there are some titles that is significantly low. So, it might be fine to merge them into one group.

# In[25]:


titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.']
def get_title(s):
    m = title_ptr.search(s)
    title = m.group()
    return title if title in titles else 'others'


# In[26]:


X['titles'] = X.Name.apply(get_title)


# In[27]:


plt.title('Titles')
sns.countplot(data=X, x='titles')


# Now, let's check the feature Fare. <br />
# There is only one person with NaN Fare. <br />
# It might not be an issue assigning the median of the fare to this sample.

# In[28]:


X[X.Fare.isnull()]


# In[29]:


fare = X[(X.Pclass==3)&(X.Embarked=='S')]['Fare'].median()
X.loc[1043,'Fare'] = fare
X.loc[1043]


# The last feature to impute is Age. <br />
# This one is somehow hard to predict. In order to keep the distribution, I'll generate random numbers between the mean. However, we'll be using the title as a guidance.

# In[30]:


sns.boxplot(data=X[~X.Age.isnull()], x="titles", y='Age')


# In[31]:


for title in np.unique(X.titles):
    qty  = len(X[(X.Age.isnull()) & (X.titles == title)]) # missing value per title
    mean = X[(~X.Age.isnull()) & (X.titles == title)]['Age'].mean() # mean of non- NaN
    std  = X[(~X.Age.isnull()) & (X.titles == title)]['Age'].std()  # std  of non- NaN
    
    rdm_age = abs(np.random.randn(qty)*std + mean) # Generating random number between the STD
    rdm_age = rdm_age.reshape(-1, 1)
    
    X.loc[X[X.Age.isnull() & (X.titles == title)].index, 'Age'] = rdm_age


# In[32]:


sns.boxplot(data=X[~X.Age.isnull()], x="titles", y='Age')


# Checking the distribution of the feature Age, we see that it didn't changed at all. <br />
# Now we don't have any missing value, so we can move on.
# 
# ---
# ### Handling some features
# While loading the safeboats, the "women and children" were the protocol. So, might be useful to bin the Age in a *child/adult* category. <br />
# Let's create a new feature with this in mind.

# In[33]:


bins = [0, 18, max(X.Age)]
categories = [1, 0]
X['is_child'] = pd.cut(X.Age, bins, labels=categories)


# The features "Sex", "title" and "Embarked" could be one-hot-encoded in order to improve the performance of the classifier.

# In[34]:


X = pd.concat([X, pd.get_dummies(X[['Sex', 'Embarked', 'titles']])], axis=1)

X['norm_fare'] = np.log(X.Fare.values+1)
# We could improve the classifier by doing a minmax operation on the feature fare.

# In[35]:


from sklearn.preprocessing import MinMaxScaler


# In[36]:


minmax = MinMaxScaler()


# In[37]:


X['norm_fare'] = minmax.fit_transform(X.Fare.values.reshape(-1,1))


# In[38]:


del X['Sex']
del X['Age']
del X['Embarked']
del X['Ticket']
del X['Name']
del X['Fare']
del X['titles']


# Now let's check our dataset...

# In[39]:


X.head()


# ---
# ## Making predictions
# Let's separate the dataset between training set and test set.

# In[40]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer


# In[41]:


X_train, X_test = X[:891], X[891:]
print('train size: {0}; test size: {1}'.format(len(X_train), len(X_test)))


# In[42]:


xtrain, xtest, ytrain, ytest = train_test_split(X_train, y)


# In[43]:


clf = RandomForestClassifier()
clf.fit(xtrain, ytrain)
pred = clf.predict(xtest)
accuracy_score(y_pred=pred, y_true=ytest)


# Without any hyperparameter tuning, the classifier with standard parameters scored an accuracy of 0.8161. <br />
# Let's try to improve this score.
# ## Tuning hyperparameters

# In[54]:


parameter_candidates = [
  {'n_estimators': [5, 10, 14, 15, 16, 20], 'criterion': ['gini', 'entropy'], \
   'random_state':[0], \
   'bootstrap':[True], \
   'min_samples_split':[2, 4, 6, 8], 'min_samples_leaf':[1, 2, 4, 6, 8], \
   'max_depth':[2, 4, 5, 6, 8, None], 'warm_start':[True, False]},

  {'n_estimators': [5, 10, 14, 15, 16, 20], 'criterion': ['gini', 'entropy'], \
   'random_state':[0], \
   'bootstrap':[False],\
   'min_samples_split':[2, 4, 6, 8], 'min_samples_leaf':[1, 2, 4, 6, 8], \
   'max_depth':[2, 4, 5, 6, 8, None],'warm_start':[True, False]}
]


# In[55]:


clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameter_candidates, n_jobs=4, scoring=make_scorer(accuracy_score))


# In[56]:


clf.fit(xtrain, ytrain)


# In[57]:


clf.best_estimator_


# In[58]:


pred = clf.predict(xtest)
accuracy_score(y_pred=pred, y_true=ytest)


# ## Conclusion
# After tuning the hyperparameters, the new classifier improved to a score of 0.820. <br />
# 
# No tuning:
#  * 0.816
#  * Submission score: 0.72727
# 
# Tuning:
#  * 0.829
#  * Submission score: 0.76555

# In[59]:


# making the submission
passengerId = np.arange(892, 1310)

pred = clf.predict(X_test)

submission = pd.DataFrame({'PassengerId':passengerId, 'Survived':pred}) 
submission.to_csv("submission.csv", index=False)

