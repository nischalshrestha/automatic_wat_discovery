#!/usr/bin/env python
# coding: utf-8

# # Yet another Titanic Kernel
# 
# Here's yet another one! Just simple logistic regression without many fancy feature generations, and regularization with parameters picked from cross-validation; really, this should be the first thing you should do with smaller datasets.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:


# Read in all of the data
all_data = pd.read_csv('../input/train.csv')
all_data.head()


# Here we're going to split everything in the following ways:
# - *Sex*: categorical with F/M being T/F
# - *Pclass:*  categorical.
# - *SibSp:* categorical, $\ge 0$ or $= 0$.
# - *Age:* kept the same.
# - *Fare:* kept the same.
# - *Embarked:* categorical.
# - *Cabin:* categorical, is NaN or not?
# 
# Everything else gets dropped from the list (yes, I didn't include anything about the names, etc!)

# In[4]:


all_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# Here, I'll assume that we observe all possible labels in the training test (this can be fixed by also appending the test set to the train set, if you're really excited, but I decided it wasn't worth it).
# 
# Also, whenever you're handling truly *big* data, you probably shouldn't do the `data.copy()` there, but rather do everything in place! (I'm getting away with a lot of stuff here...)

# In[5]:


def prepare_data(data):
    all_data = data.copy()
    
    # One-hot encode the embarked status
    one_hot_embarked = pd.get_dummies(all_data.Embarked)
    all_data = all_data.join(one_hot_embarked)
    all_data.drop('Embarked', axis=1, inplace=True)
    
    # One-hot encode the passenger class
    one_hot_pclass = pd.get_dummies(all_data.Pclass)
    all_data = all_data.join(one_hot_pclass)
    all_data.drop('Pclass', axis=1, inplace=True)
    
    # Encode if the person had any siblings
    all_data.SibSp = all_data.SibSp > 0
    
    # Encode if the person had an assigned cabin
    all_data.Cabin = all_data.Cabin.isnull()
    
    # Encode whether or not the person was female
    all_data.Sex = all_data.Sex == 'female'
    
    # Make a new category for ages which are unknown
    all_data['age_unknown'] = all_data.Age.isnull()
    all_data.replace(np.nan, 0, inplace=True)
    
    return all_data


# Separate out the survive label (to train stuff on!)

# In[6]:


all_data = prepare_data(all_data)
target = all_data.Survived
all_data.drop('Survived', axis=1, inplace=True)


# Now, we're going to perform some cross-validation to decide our hyperparameters. That is, in our Logistic Regression, we have a term (called $C$ in Sklearn's LogisticRegression class, usually in the literature, it's denoted as $1/C = \alpha$ or $1/C = \lambda$) that penalizes how complex our model is. If we make $C$ very small (i.e. $\alpha, \lambda$ very large), we force our model to be very simple, whereas making $C$ large ($\alpha, \lambda$ small) tells the optimizer that we don't care how complex our model is.
# 
# It's pretty easy to see why an overly simple model might be bad (of course, here's a super-simple model: we just predict 'survived' at every time!), but an overly-complex model is also bad. What can happen is that, the only world the classifier has seen is very small, and thus it 'overfits' to this particular set of samples we picked, predicting them perfectly. It may not be clear why we wouldn't want this, so an example is in order.
# 
# Here's a model that is extremely 'complex' [1]: for every possible passenger name, we just store if they survived or not. With this, I'm going to do very well on the training set. In fact, I'm going to do *perfectly*. But now when it comes time to test, we'll have a bunch of names I've never seen before, so we're going to do awfully on the test set.
# 
# So, we want to pick the Goldilocks zone. We don't want to make our model impossibly complex, but making it too simple won't help anyone either, so how do we go about this? It's actually (surprisingly) simple[2]! We just train our model on a portion of data (say, 80%), and pretend that the rest of the data we don't have. *Then*, after training, we test it on the other 20% of data we held in escrow. In some sense, we pretend we're Kaggle and 'score' our model privately with data it has never seen before. We do this for a bunch of possible $C$ parameters, and pick the one that does the best on the test set.
# 
# This is what we'll do below, using some niceties that sklearn brings.
# 
# ---
# 
# [1] I'm being pretty cavalier about using the term 'complex.' There are, indeed rigorous definitions of what this means, but we'll make use of the intuitive notion since I don't think adding complexity here helps (pun only partially intended).
# 
# [2] Unlike a bunch of stuff in ML. But that's okay, we gotta start somewhere.

# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


# In[8]:


n_folds = 5
kf = KFold(n_splits=n_folds, random_state=1)
best_score, best_l = -1, -1
for l in np.logspace(-5, 1, 20):
    curr_tot = 0
    for train_idx, test_idx in kf.split(all_data):
        X_train, X_test = all_data.iloc[train_idx,:], all_data.iloc[test_idx,:]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        lgr = LogisticRegression(C=l)
        lgr.fit(X_train, y_train)
        curr_tot += lgr.score(X_test, y_test)
    
    curr_tot /= n_folds
    print('Current score {} with C={}'.format(curr_tot, l))
    if curr_tot > best_score:
        best_l, best_score = l, curr_tot
        
lgr_final = LogisticRegression(C=best_l)
lgr_final.fit(all_data, target)


# We see here that picking the most complex model isn't actually the best idea; in fact, we lose out on some accuracy if we're not careful and start overfitting to our data!
# 
# Anyways, now we're ready to submit the results.

# In[9]:


test_df = pd.read_csv('../input/test.csv')
test_df = prepare_data(test_df)

# Get the PassengerIds we need for submission
output_df = test_df[['PassengerId']]

predict_df = pd.DataFrame(lgr_final.predict(test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)), dtype=np.int16)

output_df = output_df.join(predict_df, how='right')
output_df.columns = ['PassengerId', 'Survived']


# In[10]:


output_df.head()


# And that's all there is to it!
