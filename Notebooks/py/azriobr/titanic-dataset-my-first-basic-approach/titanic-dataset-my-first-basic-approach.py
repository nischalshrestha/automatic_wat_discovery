#!/usr/bin/env python
# coding: utf-8

# # Working out some data skills
# 
# After long time distant from Kaggle, I decided to take another time to practice with the new features that are being implemented in this portal. **Titanic** is a great dataset that I have just put my hands once 
# a time ago. This notebook is to naively conduct some work with scikit and test myself in the competitions again, but for now, only small codes and little descriptions. If someone have any critics, please feel 
# free to comment in the section below.
# 
# I pretend to do a little improvement in the usage of cross_validation with the mainstream algorithms, like **SVM**, **RandomForests** and **XGBoost**.
# 
# Hope you enjoy it!

# In[135]:


import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
#Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
#from sklearn import feature_selection as fs
#from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')


# 
# First things, first
# 
# ## Reading data

# In[136]:


train_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
train_set.shape, test_set.shape


# Let's take a look into the data. First, we concatenate train and test sets, then we some basic plots:

# ## First look at datasets

# Let's take a look into the data

# In[137]:


full_set = pd.concat([train_set, test_set])
full_set.head()


# In[138]:


full_set.info()


# 
# Above we can see that we have to work with mixed data. Some important columns like **Age** and **Cabin** have missing data. Another good information to be taken into account is the **Pclass** , which tells us the social class of the person. As stated in many documentaries/movies about Titanic sinking, many people who survived where likely to be in the upper class, to be women or child. This way, a good approach may have to consider **Sex** information.
# 
# Let's see some numbers before making imputations and some data cleaning/pre-processing:

# In[139]:


full_set.describe()


# From the table above we see that around 38% of the passengers in the train dataset have survived. As stated in [https://www.kaggle.com/c/titanic](), 1502 of 2224 (passengers and crew members) haven't survived, around 32%.
# It's shown the range for **Age**, which spans from 0 years old to 80. It's interesting to see the discrepancy among the **Fare** values, where it's median is around 14.5 while its maximum goes around 512.

# ## Feature Engineering

# In[140]:


full_set.head(10)


# In[141]:


# Age
full_set['Age'][full_set['Age'].isnull()] = full_set['Age'].median()
full_set['Age'] = full_set['Age'].astype(int)


# In[142]:


# Cabin is a feature that could be used to map the passengers to their locations inside the ship and create some levels 
# of difficulty in acessing the deck, and the lifeboats, however to this dataset one simpler approach is to configure
# if someone has a cabin or not, since many people didn't have one.
full_set['HasCabin'] = full_set['Cabin'].apply(lambda x: 0 if isinstance(x, float) else 1)


# In[143]:


full_set.Embarked.unique()


# In[144]:


full_set.Embarked.value_counts().plot.bar()


# In[145]:


# Embarked will be imputed by the modal, the category S
full_set['Embarked'][full_set['Embarked'].isnull()] = 'S'
full_set['Embarked'] = full_set['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[146]:


# Fare seems ok at first, may be try to categorize it later..
full_set[ full_set['Fare'].isnull() ]


# In[147]:


# As just one man haven't an informed Fare, just imputes with 0.
full_set['Fare'][ full_set['Fare'].isnull() ] = 0.


# In[148]:


# Name seems to not influence that much at first, but as stated by the Kaggle community, some new feature can be derived from it
# which may be helpful. Since it contains persons titles, we may infer something about social class too
def get_title(string):
    result = re.search(' ([A-Za-z]+)\.', string)
    if result:
        return result.group(1)
    else:
        return ""


# In[149]:


full_set['Title'] = full_set['Name'].apply(get_title)
print(full_set['Title'].unique())


# In[150]:


commons = ['Mr','Mrs','Miss','Mme','Ms','Mlle']
rares = list(set(full_set['Title'].unique()) - set(commons))
full_set['Title'] = full_set['Title'].replace('Ms','Miss')
full_set['Title'] = full_set['Title'].replace('Mlle','Miss')
full_set['Title'] = full_set['Title'].replace('Mme','Mrs')
full_set['Title'][full_set['Title'].isin(rares)] = 'Rare'
full_set['Title'] = full_set['Title'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Rare':3})


# In[151]:


#It is possible to aggregate the features Parch and SibSp to compose another variable to measure the "family" of the passenger
full_set['FamSize'] = full_set['Parch'] + full_set['SibSp'] + 1 
# The plus 1 above refers to the person itself


# In[152]:


# Sex
full_set['Sex'] = full_set['Sex'].map({'male':0, 'female':1})


# In[153]:


full_set.drop(['Cabin','Name','Parch','PassengerId','SibSp','Ticket'], axis=1, inplace=True)


# Now we gost a much cleaner dataset:

# In[154]:


full_set.head()


# Let's do some plots:

# In[155]:


# Borrowed from http://stackoverflow.com/a/31385996/4099925
def hexbin(x, y, color, max_series=None, min_series=None, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    ax = plt.gca()
    xmin, xmax = min_series[x.name], max_series[x.name]
    ymin, ymax = min_series[y.name], max_series[y.name]
    plt.hexbin(x, y, gridsize=15, cmap=cmap, extent=[xmin, xmax, ymin, ymax], **kwargs)


# In[156]:


train_part = full_set[:train_set.shape[0]]
g = sns.PairGrid(full_set[:train_set.shape[0]], hue='Survived')
g.map_diag(plt.hist)
g.map_lower(plt.scatter, alpha=0.4)
g.map_upper(hexbin, min_series=train_part.min(), max_series=train_part.max(), alpha=0.5)


# We see one expected behavior here, with females having higher rates of survival in all classes. Another observation is the higher survival rates for ladies with title 'Miss', whose age also agree to an expected range.
# 
# Now, let's swim into some machine learning.

# ## Trying something a bit better

# In[157]:


full_set['Fare'].quantile(np.linspace(0,1,5))


# In[158]:


def fare_bin(fare):
    if fare <= 7.8958:
        return 0.
    elif 7.8958 < fare <= 14.4542:
        return 1.
    elif 14.4542 < fare <= 31.2750:
        return 2.
    else:
        return 3.


# In[159]:


full_set['Fare'] = full_set['Fare'].apply(fare_bin).astype(int)


# In[161]:


full_set.head()


# In[176]:


full_set['Pclass'] = full_set['Pclass'].astype('str')


# In[177]:


full_w_dum = pd.get_dummies(full_set)
full_w_dum.head()


# In[178]:


full_set = full_w_dum


# In[179]:


cols = list(set(full_set.columns) - set(['Survived']))
X_train, X_test = full_set[:train_set.shape[0]][cols], full_set[train_set.shape[0]:][cols]
y_train = full_set[:train_set.shape[0]]['Survived']


# In[180]:


# Running some models, testing with small cross_validation and f1 metric (binary target) and running the prediction
# with all the training set
models = [ LogisticRegression, SVC, LinearSVC, RandomForestClassifier, KNeighborsClassifier, XGBClassifier ]
mscores = []
lscores = ['f1','accuracy','recall','roc_auc']
np.random.seed(42) # Reproducibility of results is very important!
for elem in models:
    mscores2 = []
    model = elem()
    for sc in lscores:
        scores = cross_val_score(model, X_train, y_train, scoring=sc)
        mscores2.append(np.mean(scores))
    mscores.append(mscores2)


# In[181]:


order = np.argsort(np.mean(np.array(mscores), axis=1))
print(order)


# ## Checking with some k-Fold cross-validation 

# In[182]:


from sklearn.model_selection import StratifiedKFold, KFold


# In[183]:


# K-Fold
results_kfold = []
for K in [5,6,7,8,9,10]:
    model = XGBClassifier()
    kfold = KFold(n_splits=K, random_state=42)
    res = cross_val_score(model, X_train, y_train, cv=kfold)
    results_kfold.append((K,res.mean()*100, res.std()*100))


# In[184]:


list(map(lambda x: print('Iteration nº {:2}, with acc. {:.12} and std. dev. {:.12}'.format(*x)),results_kfold));


# ## Checking with some cross-validation using stratified k-fold

# In[185]:


# Stratified K-Fold
results_strat_kfold = []
for K in [5,6,7,8,9,10]:
    model = XGBClassifier()
    kfold = StratifiedKFold(n_splits=K, random_state=42)
    res = cross_val_score(model, X_train, y_train, cv=kfold)
    results_strat_kfold.append((K,res.mean()*100, res.std()*100))


# In[186]:


list(map(lambda x: print('Iteration nº {:2}, with acc. {:.12} and std. dev. {:.12}'.format(*x)),results_strat_kfold));


# In[170]:


ax = y_train.plot.hist()
ax.set_title('Frequency of survivorship in training set')


# Now, for XGBoost, using CV to better tuning the model

# In[187]:


xg_scores = []
kfold = StratifiedKFold(n_splits=7, random_state=42)
for lamb in [ .05, .1, .2, .3, .4, .5, .6 ]:
    for eta in [.2, .19, .17, .15, .13, .11]:
        model = XGBClassifier(learning_rate=eta, reg_lambda=lamb)
        res = cross_val_score(model, X_train, y_train, cv=kfold)
        xg_scores.append({'lamb':lamb, 'eta':eta, 'acc':res.mean()})


# Then we pick the one with best accuracy:

# In[188]:


sorted(xg_scores,key=lambda x: x['acc'], reverse=True)[0]


# Train the final model

# In[189]:


model = XGBClassifier(learning_rate=.17, reg_lambda=.5)
model.fit(X_train, y_train)
predicted = model.predict(X_test)


# In[190]:


test_set['Survived'] = predicted.astype(int)
test_set[['PassengerId','Survived']].to_csv('submission.csv', sep=',', index=False)


# The readings below were performed to check the output file format

# In[191]:


get_ipython().run_cell_magic(u'bash', u'', u'head -10 submission.csv')


# In[ ]:


get_ipython().run_cell_magic(u'bash', u'', u'head -10 ../input/gender_submission.csv')


# # Resume
# I could submit some versions with little accuracy in the leaderboard, at moment **5481st** position, using linearSVM as the best model, 
# chosen after small iterations of cross validation, with no further details in the parameters setup. I expect to have better result with a good 
# fine tuning using XGBoost and RandomForests later. 
# 
# I'm very grateful to the Kagglers who have put great effort in some amazing and well designed basic tutorials. These starters were of nice 
# help for me.

# In[ ]:




