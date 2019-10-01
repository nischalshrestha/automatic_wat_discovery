#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster
# 
# ***By Joe Corliss***

# ## Table of Contents

# * [1 Introduction](#1)
# * [2 Getting Started](#2)
#     * [2.1 Imports](#2.1)
#     * [2.2 Read In the Data](#2.2)
# * [3 Pre-processing](#3)
#     * [3.1 Feature Engineering](#3.1)
#     * [3.2 One-Hot Encoding](#3.2)
#     * [3.3 Imputation with Mean Substitution](#3.3)
#     * [3.4 Standardization](#3.4)
#     * [3.5 Recursive Feature Elimination](#3.5)
# * [4 Predictive Modeling](#4)
#     * [4.1 Random Forest](#4.1)
#     * [4.2 Gradient Boosting](#4.2)
#     * [4.3 Logistic Regression](#4.3)
#     * [4.4 Gaussian Naive Bayes](#4.4)
#     * [4.5 Support Vector Classifier](#4.5)
#     * [4.6 k-Nearest Neighbors](#4.6)
# * [5 Conclusion](#5)
#     * [5.1 Results Summary](#5.1)
#     * [5.2 Test Set Predictions](#5.2)

# # 1 Introduction
# <a id='1'></a>

# [Kaggle competition](https://www.kaggle.com/c/titanic) - The competition page on Kaggle
# 
# [Kaggle notebook](https://www.kaggle.com/pileatedperch/titanic-predicting-survival) - This notebook hosted on Kaggle
# 
# [GitHub repository](https://github.com/jgcorliss/titanic-competition) - GitHub repo for this project
# 
# "The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# Your score is the percentage of passengers you correctly predict. This is known simply as 'accuracy.'"

# # 2 Getting Started
# <a id='2'></a>

# ## 2.1 Imports
# <a id='2.1'></a>

# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas options
pd.set_option('display.max_colwidth', 1000, 'display.max_rows', None, 'display.max_columns', None)

# Plotting options
get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set(style='whitegrid')


# ## 2.2 Read In the Data
# <a id='2.2'></a>

# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[3]:


df_train.shape


# In[4]:


df_test.shape


# Save the test set passenders IDs for later:

# In[5]:


TestPassengerId = df_test.loc[:,'PassengerId']


# Concatenate the train and test sets together:

# In[6]:


df = df_train.append(df_test, ignore_index=True)


# Basic metadata:

# In[7]:


df.shape


# In[8]:


df.info()


# Find incomplete columns:

# In[9]:


def incomplete_cols(df):
    """
    Returns a list of incomplete columns in df and their fraction of non-null values.
    
    Input: pandas DataFrame
    Returns: pandas Series
    """
    cmp = df.notnull().mean().sort_values()
    return cmp.loc[cmp<1]


# In[10]:


incomplete_cols(df)


# In[11]:


df.sample(5) # Display some random rows


# # 3 Pre-processing
# <a id='3'></a>

# ## 3.1 Feature Engineering
# <a id='3.1'></a>

# ### Age

# In[12]:


df['Age'].notnull().mean()


# In[13]:


df['Age_NA'] = df['Age'].isnull()


# In[14]:


plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Age'].notnull(), 'Age'], bins=range(0, 90, 2), kde=False)
plt.ylabel('Count')
plt.title('Histogram of Passenger Age')


# ### Cabin

# In[15]:


df['Cabin'].notnull().mean()


# Extract the cabin letter (A, B, C, etc), or NA if the cabin data is missing.

# In[16]:


def find_cabin(s):
    try:
        return s[0]
    except:
        return 'NA'


# In[17]:


df.loc[:,'Cabin'] = df['Cabin'].apply(find_cabin)


# In[18]:


df['Cabin'].value_counts()


# ### Embarked

# In[19]:


df['Embarked'].value_counts(dropna=False)


# Only 2 passengers do not have a port of embarkation.

# In[20]:


sns.countplot(x='Embarked', data=df)
plt.title('Passenger Ports of Embarkation')


# ### Fare

# In[21]:


df['Fare'].isnull().sum()


# Only one fare is missing. Who is that?

# In[22]:


df.loc[df['Fare'].isnull()]


# Not sure why his fare is missing. Let's plot the fares:

# In[23]:


plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Fare'].notnull(), 'Fare'], kde=False)
plt.ylabel('Count')
plt.title('Histogram of Passenger Fares')


# In[24]:


df['Fare'].skew()


# The fares are right-skewed.

# ### Name

# In[25]:


df['Name'].notnull().mean()


# In[26]:


df['Name'].sample(5)


# Let's extract everyone's titles.

# In[27]:


df['Title'] = df['Name'].apply(lambda s: s.split(', ')[1].split(' ')[0])


# In[28]:


df['Title'].nunique()


# Did it work?

# In[29]:


df[['Name', 'Title']].sample(10)


# Seems good. Title value counts:

# In[30]:


df['Title'].value_counts()


# The only odd-looking value is "the". Let's investigate.

# In[31]:


df.loc[df['Title']=='the']


# Her title is actually "Countess."

# There doesn't seem to be any new information in `Title`. We already know the passenger age, sex, marital status (`SibSp`), and economic class (`Pclass`). Furthermore, many of the titles have too few data points to be useful. So we're going to drop the `Name` and `Title` columns.

# In[32]:


df.drop(labels=['Name','Title'], axis=1, inplace=True)


# ### Parch

# In[33]:


df['Parch'].notnull().mean()


# In[34]:


df['Parch'].value_counts()


# In[35]:


plt.figure(dpi=80)
sns.countplot(x='Parch', data=df)
plt.title('Number of Parents/Children')


# ### PassengerId

# In[36]:


df.shape[0]


# In[37]:


df['PassengerId'].nunique()


# All passenger IDs are unique. The IDs will (probably) be removed in automatic variable selection.

# ### Pclass

# In[38]:


df['Pclass'].notnull().mean()


# In[39]:


df['Pclass'].value_counts()


# A majority of the passengers are Lower Class.

# In[40]:


plt.figure(figsize=(4,4), dpi=90)
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Ticket Class')


# ### Sex

# In[41]:


df['Sex'].notnull().mean()


# In[42]:


df['Sex'].value_counts(normalize=True)


# The passengers are 64.4% male.

# ### SibSp

# In[43]:


df['SibSp'].notnull().mean()


# In[44]:


df['SibSp'].value_counts()


# In[45]:


plt.figure(dpi=90)
sns.countplot(x='SibSp', data=df)
plt.title('Number of Siblings/Spouses')


# ### Survived

# This is the target variable.

# In[46]:


df.loc[df['Survived'].notnull(), 'Survived'].value_counts(normalize=True)


# So there's a 38.4% survival rate in the training set.

# ### Ticket

# In[47]:


df['Ticket'].notnull().mean()


# In[48]:


(df['Ticket'].nunique(), df.shape[0])


# Apparently there are duplicate tickets...? That's weird.

# In[49]:


df['Ticket'].sample(10)


# Maybe the prefix on some of the tickets is important? Let's grab it.

# In[50]:


def ticket_prefix(s):
    'Find the content of the ticket before the ticket number'
    temp = s.split(' ')
    if len(temp) > 1:
        return ' '.join(temp[:-1])
    else:
        return 'NONE'


# In[51]:


df.loc[:,'Ticket Prefix'] = df['Ticket'].apply(ticket_prefix)


# In[52]:


df['Ticket Prefix'].nunique()


# In[53]:


df['Ticket Prefix'].value_counts()


# Some of the prefixes are very similar. Let's assume that the characters `.`, `/`, and whitespace and not significant, and that `SC/PARIS` is the same as `SC/Paris`.

# In[54]:


df.loc[:,'Ticket Prefix'] = df['Ticket Prefix'].apply(lambda s: s.replace('.','').replace('/','').replace(' ','').upper())


# In[55]:


df['Ticket Prefix'].nunique()


# In[56]:


df['Ticket Prefix'].value_counts()


# If there are fewer than 10 occurences of a particular prefix, we'll reclassify it as "other."

# In[57]:


vals = df['Ticket Prefix'].value_counts()

def other_prefix(prefix):
    if vals[prefix] < 10:
        return "OTHER"
    else:
        return prefix


# In[58]:


df.loc[:,'Ticket Prefix'] = df['Ticket Prefix'].apply(other_prefix)


# In[59]:


df['Ticket Prefix'].value_counts()


# Now let's extract the ticket number. As it turns out, a very small number of tickets have no ticket number, so we'll assign NaN for those tickets.

# In[60]:


def ticket_number(s):
    'Find the ticket number on a ticket'
    try:
        return np.int64(s.split(' ')[-1])
    except:
        return np.nan


# In[61]:


df['Ticket Number'] = df['Ticket'].apply(ticket_number)


# In[62]:


df['Ticket Number'].isnull().sum()


# There are 4 tickets with no ticket number. Let's look at those:

# In[63]:


df.loc[df['Ticket Number'].isnull()]


# Their ticket just says "LINE." What does that mean?

# Let's plot the ticket numbers.

# In[64]:


plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Ticket Number'].notnull(), 'Ticket Number'], bins=400, kde=False)
plt.ylabel('Count')
plt.title('Histogram of Ticket Number')


# In[65]:


plt.figure(figsize=(12,4), dpi=90)
sns.distplot(df.loc[df['Ticket Number'].notnull(), 'Ticket Number'], bins=500, kde=False)
plt.xlim([0, 500000])
plt.ylabel('Count')
plt.title('Histogram of Ticket Number')


# There are 5 "classes" of ticket numbers. Let's convert the ticket numbers into these categorical classes.

# In[66]:


def tick_num_class(tick_num):
    if tick_num < 100000:
        return 'A'
    elif tick_num < 200000:
        return 'B'
    elif tick_num < 300000:
        return 'C'
    elif tick_num < 400000:
        return 'D'
    elif tick_num >= 400000:
        return 'E'
    else:
        return 'NA'


# In[67]:


df.loc[:,'Ticket Number'] = df['Ticket Number'].apply(tick_num_class)


# In[68]:


df['Ticket Number'].value_counts()


# In[69]:


df.drop(labels='Ticket', axis=1, inplace=True)


# ## 3.2 One-Hot Encoding
# <a id='3.2'></a>

# Convert categorical features to binary features via one-hot encoding.

# In[70]:


df.sample(5)


# In[71]:


df.info()


# Five columns will be one-hot encoded: `Cabin`, `Embarked`, `Sex`, `Ticket Prefix`, and `Ticket Number`. Are any of these columns incomplete?

# In[72]:


incomplete_cols(df)


# Only `Embarked` is. But only 2 values were missing in that column, so we won't create columns for NA values.

# In[73]:


df.shape


# In[74]:


df = pd.get_dummies(df, drop_first=True)


# In[75]:


df.shape


# In[76]:


df.sample(5)


# ## 3.3 Imputation with Mean Substitution
# <a id='3.3'></a>

# In[77]:


X = df.drop(labels='Survived', axis=1)
y = df.loc[:,'Survived']


# In[78]:


incomplete_cols(X)


# In[79]:


from sklearn.preprocessing import Imputer


# In[80]:


imputer = Imputer().fit(X)


# In[81]:


X = pd.DataFrame(imputer.transform(X), columns=X.columns)


# In[82]:


incomplete_cols(X)


# ## 3.4 Standardization
# <a id='3.4'></a>

# Transform all the features to zero mean and unit variance.

# In[83]:


from sklearn.preprocessing import StandardScaler


# In[84]:


scaler = StandardScaler().fit(X)


# In[85]:


X = pd.DataFrame(scaler.transform(X), columns=X.columns)


# In[86]:


X.sample(5)


# ## 3.5 Recursive Feature Elimination
# <a id='3.5'></a>

# Train/test split:

# In[87]:


X_train = X.loc[y.notnull()]
X_test = X.loc[y.isnull()]
y_train = y[y.notnull()].apply(np.bool)


# Recursive feature elimination using a Linear Support Vector Classifier:

# In[88]:


from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV


# In[89]:


selector = RFECV(estimator=LinearSVC(), scoring='accuracy', verbose=0, n_jobs=-1)


# In[90]:


selector.fit(X_train, y_train)


# How many features were retained?

# In[91]:


selector.n_features_


# Which features were retained?

# In[92]:


X_train.columns[selector.support_]


# Score of the underlying LinearSVC on the training set:

# In[93]:


selector.score(X_train, y_train)


# Hopefully there was not too much overfitting.

# Reduce our data to the retained features:

# In[94]:


X_train = X_train.loc[:,selector.support_]
X_test = X_test.loc[:,selector.support_]


# # 4 Predictive Modeling
# <a id='4'></a>

# In[95]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer


# ## 4.1 Random Forest
# <a id='4.1'></a>

# In[96]:


from sklearn.ensemble import RandomForestClassifier


# In[97]:


param_grid = {'n_estimators': [10, 55, 100],
              'max_features': ['log2', 'sqrt', None],
              'max_depth': [5, 15, 30, None],
              'min_samples_split': [2, 10, 50],
              'min_samples_leaf': [1, 5, 10]
             }


# In[98]:


model_rfc = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)


# In[99]:


model_rfc.fit(X_train, y_train)


# In[100]:


model_rfc.best_params_


# In[101]:


model_rfc.best_score_


# Feature importances:

# In[102]:


plt.figure(figsize=(4,4), dpi=90)
sns.barplot(y=X_train.columns, x=model_rfc.best_estimator_.feature_importances_, color='darkblue', orient='h')
plt.xlabel('RF Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importances')


# ## 4.2 Gradient Boosting
# <a id='4.2'></a>

# In[103]:


from sklearn.ensemble import GradientBoostingClassifier


# In[104]:


param_grid = {'max_depth': [3, 12, 25],
              'subsample': [0.6, 0.8, 1.0],
              'max_features': [None, 'sqrt', 'log2']
             }


# In[105]:


model_gb = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)


# In[106]:


model_gb.fit(X_train, y_train)


# In[107]:


model_gb.best_params_


# In[108]:


model_gb.best_score_


# ## 4.3 Logistic Regression
# <a id='4.3'></a>

# In[109]:


from sklearn.linear_model import LogisticRegression


# In[110]:


param_grid = {'penalty': ['l2'],
              'C': [10**k for k in range(-3,3)],
              'class_weight': [None, 'balanced'],
              'warm_start': [True]
             }


# In[111]:


model_logreg = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)


# In[112]:


model_logreg.fit(X_train, y_train)


# In[ ]:


model_logreg.best_params_


# In[ ]:


model_logreg.best_score_


# ## 4.4 Gaussian Naive Bayes
# <a id='4.4'></a>

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


model_gnb = GaussianNB()


# In[ ]:


model_gnb.fit(X_train, y_train)


# In[ ]:


accuracy_score(y_train, model_gnb.predict(X_train))


# ## 4.5 Support Vector Classifier
# <a id='4.5'></a>

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


param_grid = {'C': [10**k for k in range(-3,4)],
              'class_weight': [None, 'balanced'],
              'shrinking': [True, False]
             }


# In[ ]:


model_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)


# In[ ]:


model_svc.fit(X_train, y_train)


# In[ ]:


model_svc.best_params_


# In[ ]:


model_svc.best_score_


# ## 4.6 k-Nearest Neighbors
# <a id='4.6'></a>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


param_grid = {'n_neighbors': [1, 2, 4, 8, 16, 32, 64, 128, 256],
              'weights': ['uniform', 'distance'],
              'p': [1, 2, 3, 4, 5]
             }


# In[ ]:


model_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring=make_scorer(accuracy_score), n_jobs=-1, verbose=1)


# In[ ]:


model_knn.fit(X_train, y_train)


# In[ ]:


model_knn.best_params_


# In[ ]:


model_knn.best_score_


# # 5 Conclusion
# <a id='5'></a>

# ## 5.1 Results Summary
# <a id='5.1'></a>

# In[ ]:


print('Training Accuracy Scores')
print('Random Forest: ', model_rfc.best_score_)
print('Gradient Boosting: ', model_gb.best_score_)
print('Logistic Regression: ', model_logreg.best_score_)
print('Gaussian Naive Bayes: ', accuracy_score(y_train, model_gnb.predict(X_train)))
print('Support Vector Classifier: ', model_svc.best_score_)
print('k-Nearest Neighbors: ', model_knn.best_score_)


# ## 5.2 Test Set Predictions
# <a id='5.2'></a>

# In[ ]:


y_preds = model_rfc.predict(X_test)


# In[ ]:


submission = pd.DataFrame({'PassengerId':TestPassengerId, 'Survived':np.uint8(y_preds)})


# In[ ]:


submission.shape


# In[ ]:


submission.sample(5)


# In[ ]:


submission.to_csv('my_submission.csv', index=False)

