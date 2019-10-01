#!/usr/bin/env python
# coding: utf-8

# # A beginners take to logistic regression and the titanic dataset.
# 
# This notebook follows the logistic regression section from Jose Portilla's amazing [Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/content).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns # easy visualization
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# load the test and train sets, concat them together for cleaning and feature engineering
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_full = pd.concat([df_train, df_test], axis = 0, ignore_index=True)

# save the test PassengerID for future use
passengerId_test = df_test['PassengerId']
df_full.head()


# ## Exploratory analysis
# 
# Let's take a look at a summary of this dataframe.

# In[ ]:


df_full.describe()


# In[ ]:


df_full.info()


# Looks like we have 1309 total observations, 891 of which have the survived label. We are missing some fare information and a decent amount of age information. We will have to deal with that later. Let's check this visually.

# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(df_full.isnull(), yticklabels=False, cbar=False)


# We are also missing a lot of cabin information! We will have to deal with these missing values later. Let's explore how many people survived the disaster in general and as it relates to different features.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=df_full)


# In[ ]:


sns.countplot(x='Survived', hue='Sex', data=df_full)


# More people died than survived in the training set. Of those that survived the ratio of female to male is two to one. Of those that died it was much more men than women.

# In[ ]:


sns.countplot(x='Survived', hue='Pclass', data=df_full)


# The passenger class is somewhat evenly distributed in terms of survivors, but of those that perished, odds were they were in the 3rd class.

# In[ ]:


sns.distplot(df_full['Age'].dropna(), kde=False, bins = 30)


# Looks like there were a lot of very small children on the Titanic, and then a decent amount of people in their 20s and 30s.

# In[ ]:


sns.countplot(x='SibSp', data = df_full)


# Most people on board are singletons! Second most common type of passenger is someone who has either one spouse or one sibling.
# 

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(df_full['Fare'].dropna(), rug=True, hist=False)


# ## Cleaning and imputation
# 
# Let's take care of all of those missing values in the age column.
# 
# We could just put in the average age for each passenger, but that isn't elegant. Instead we can look at the missing age for each passenger class.

# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=df_full)


# Looks like first class passengers are older than the second and first class passengers. This makes sense, wealth takes time to build!
# 
# Now we can create a model to estimate the age of all of our passengers. But instead we will just put in the average age for each class and slate fancier imputation for the future.

# In[ ]:


df_full.groupby('Pclass').mean()['Age']


# In[ ]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 39.159930
        elif Pclass == 2:
            return 29.506705
        else:
            return 24.816367
    
    else:
        return Age


# In[ ]:


# now apply this function
df_full['Age'] = df_full[['Age', 'Pclass']].apply(impute_age, axis = 1)


# Let's check to ensure that the missing values were filled in.

# In[ ]:


df_full.info()


# Finally, we have two missing values in `Embarked` and one in `Fare`. We can fill in the missing fare by looking how it relates to Passenger Class.

# In[ ]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Fare', data=df_full)


# In[ ]:


df_full.groupby('Pclass').mean()['Fare']


# Let's find the missing fare in the data.

# In[ ]:


df_full[df_full['Fare'].isnull()]


# Since Mr. Thomas Storey is a 3rd class passenger, let's assign the average fare of $13.30.

# In[ ]:


df_full.loc[df_full['PassengerId'] == 1044, 'Fare'] = 13.302889
df_full[df_full['PassengerId'] == 1044]


# Perfect. Now on to `Embarked`. Where do we have missing data?

# In[ ]:


df_full[df_full['Embarked'].isnull()]


# Looks like these passengers share the same ticket information and were both singletons travelling first class. Following the imputation analysis [here](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic), it appears they departed from Charbourg.

# In[ ]:


df_full.loc[df_full['PassengerId'].isin([62, 830]), 'Embarked'] = 'C'
df_full.loc[df_full['PassengerId'].isin([62, 830])]


# In[ ]:


df_full.info()


# We have filled in all our missing data! We will deem `Cabin` a lost cause at this point and we won't investigate the `Ticket` column, so let's drop those here.

# In[ ]:


df_full.drop(['Cabin', 'Ticket'], axis=1, inplace=True)


# In[ ]:


df_full.info()


# Now time to deal with categorical features. We need to create a dummy variable to turn `Sex` and `Embarked` into 0s and 1s.

# In[ ]:


sex = pd.get_dummies(df_full['Sex'], drop_first=True)
sex.head()


# In[ ]:


embark = pd.get_dummies(df_full['Embarked'], drop_first=True)
embark.head()


# In[ ]:


df_full = pd.concat([df_full, sex, embark], axis = 1)
df_full.drop(['Sex', 'Embarked'], axis = 1, inplace=True)
df_full.head()


# And finally, in this simple kernel we won't investigate a passenger's name nore their PassengerId.

# In[ ]:


df_full.drop(['Name', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


df_full.head()


# Note that while Passenger Class is categorical, it's numerical value has meaning, so we are currently treating it as continuous. However, this could be explored to potentially improve our model.
# 
# ## Building the logistic regression model
# 
# Now that we have our data prepared, I am going to split `df_full` back into training and test sets, and I'm going to estimate the accuracy of our model on the test set by implementing [Stratified k-fold cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) on the training set.

# In[ ]:


df_train = df_full[:891]
df_test = df_full[891:]


# In[ ]:


X = df_train.drop('Survived', axis=1)
y = df_train['Survived']


# In[ ]:


# sklearn imports
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[ ]:


skf = StratifiedKFold(y, n_folds=3)


# In[ ]:


for train_index, test_index in skf:
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    logmodel = LogisticRegression()
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test, predictions))


# Looks like we can expect a model accuracy of about 80%. Not bad for just a little bit of work! Let's go ahead and make predictions on the test set using the entirety of the training set to build our model.

# In[ ]:


X_test = df_test.drop('Survived', axis=1)
logmodel = LogisticRegression()
logmodel.fit(X, y)
predictions = logmodel.predict(X_test)


# In[ ]:


df_predictions = pd.DataFrame({'PassengerID' : passengerId_test, 'Survived' : predictions.astype(int)})
df_predictions.to_csv('logistic_regression_submission.csv', index=False)

