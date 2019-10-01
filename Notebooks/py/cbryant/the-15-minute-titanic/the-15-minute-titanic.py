#!/usr/bin/env python
# coding: utf-8

# # Quick Look and Code 

# This notebook is not about getting a top score on the leaderboard. It isn't about feature engineering or using multiple ML techniques. There are plenty of discussions and kernels to do that. This is about getting a quick understanding and spitting out useable results in less than 30 min. Sometimes in real life, that's all you get! 

# Those of you working in data science have likely come to a point where someone has said - can this work? Can you get me information about this in a day? or an hour? This becomes about quickly sizing up the data, using not just the visualizations of the data itself but a quick understanding of it in your gut (and maybe proof via some simple code).
# You tell your boss "Here, I got somewhere and yes the data seems to say something BUT this is not even close to perfect. It simply shows you it's possible." Whether or not she believes you is how you present it.

# ## Take a look

# First, grab the csv via pandas and print out a few rows.

# In[ ]:


import pandas as pd
import numpy as np #I know I don't need this yet but I always call it anyway since I invariably use it at some point

train = pd.read_csv('../input/train.csv')
train.info()
train.head(10)


# Okay - so, there are 7 comlumns that are numerical. Bonus! Let's ignore the others for now. We can see that there is at least 1 NaN in age. Let's double check and see how many others:

# In[ ]:


train.isnull().sum()


# Great - looks like Age has about 20% NaN. Well, we could do a few things about this. Since I think that a random forest is likely going to provide the fastest and simplest anwser, maybe setting them to zero or removing them from the set would work. Since 20% is a lot, it might be easiest to set them to zero for now. Probably the poorest solution but we are thinking fast right now. Also, let's store it under a different variable just in case.

# In[ ]:


train['Age_0'] = train['Age'].fillna(0)
train['Age_0'].head(10)


# There - we see the NaN was switched - double check everything else:

# In[ ]:


train.isnull().sum()


# Great! We don't need the "PassengerId" now - it is unique. We want to find out who survived, so our target value is "Survived". Let's build out a quick and dirty training set:

# In[ ]:


Ytrain = train['Survived']
Xtrain = np.array(train.iloc[:,[2,6,7,9,12]])


# Okay - previous experience tells us - we better run our random forest with some folds - lets say 5 - and let's shuffle the data too. There are likely issues here with leakage but let's ignore them for now.

# In[ ]:


from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)


# let's get our RF classifier prepped.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf 


# Perfect. You know what? I just cut that chunk of code from another piece of software I wrote. Saved me time! In fact - I do that a lot. Glad I remembered...
# 
# But, uh, I should grab my test data first I guess... Almost forgot!

# In[ ]:


test = pd.read_csv('../input/test.csv')
test.info()
test.head(5)
test.isnull().sum()


# Uh-oh - looks like we have a problem. Our test data has some NaN in Age too... and 1 in Fare! Let's through those to zero too for now.

# In[ ]:


test['Age_0'] = test['Age'].fillna(0)
test['Fare_0'] = test['Fare'].fillna(0)
test.isnull().sum()


# In[ ]:


Xtest = np.array(test.iloc[:,[1,5,6,11,12]])


# Okay - run through the folds, train each model, predict some stuff, looks at metrics, and then out put the test predictions for each fold and store them. 

# In[ ]:


i=0
pred = pd.DataFrame({'PassengerId': test['PassengerId']})

from sklearn import metrics

for train_index, cv_index in kf.split(Xtrain, Ytrain):
    Xtrain_K, X_cv = Xtrain[train_index], Xtrain[cv_index]
    Ytrain_K, Y_cv = Ytrain[train_index], Ytrain[cv_index]      
    
    trained_model = random_forest_classifier(Xtrain_K, Ytrain_K)
    

    pred_tr = trained_model.predict(Xtrain.reshape(len(Ytrain),-1))

    tr_acc = metrics.accuracy_score(Ytrain, pred_tr)
    f1_sc = metrics.f1_score(Ytrain, pred_tr)
    
    print('F1-score: ',f1_sc)
    print('Accuracy score: ',tr_acc)
    
    pred_test = trained_model.predict(Xtest)
    
    pred['pred_'+str(i)] = pred_test
    i=i+1


# Great - Ran our 5 folds and stored them... let's take a look:

# In[ ]:


pred.head(5)


# Looks a bit messy. More folds would probably help, but we can look at that later. F1-score looks okay and accuracy isn't horrible. Not bad for a quick 5 minute chunk of code. I ensembled this via majority voting and submitted it.

# In[ ]:


fields = pred.iloc[:, 1:]
pred['Survived'] = fields.mode(axis=1)
pred[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)


# Bam! 0.622 - about 9450th on the leaderboard. Only 5 features with the missing values poorly engineered. But, let's think. We all have heard about the sinking of the Titanic (maybe even seen the movie). Who tends to survive? Well, Rose in the movie right? Given the early 20th century, men tended to try and save the women (or women and children). So, let's include the sex of the people in the code.

# In[ ]:


idx_mf = (np.where(train['Sex'] == 'female'))[0]
sex = np.zeros(len(Ytrain))
sex[idx_mf] = 1 # set to 1 for female otherwise leave as zero for male
Xtrain = np.hstack((Xtrain, sex.reshape((len(sex),-1))))

idx_mf = (np.where(test['Sex'] == 'female'))[0]
test_sex = np.zeros(len(test['Sex']))
test_sex[idx_mf] = 1 # set to 1 for female otherwise leave as zero for male
Xtest = np.hstack((Xtest, test_sex.reshape((len(test_sex),-1))))


# Now, run the RF again:

# In[ ]:


for train_index, cv_index in kf.split(Xtrain, Ytrain):
    Xtrain_K, X_cv = Xtrain[train_index], Xtrain[cv_index]
    Ytrain_K, Y_cv = Ytrain[train_index], Ytrain[cv_index]      
    
    trained_model = random_forest_classifier(Xtrain_K, Ytrain_K)
    

    pred_tr = trained_model.predict(Xtrain.reshape(len(Ytrain),-1))

    tr_acc = metrics.accuracy_score(Ytrain, pred_tr)
    f1_sc = metrics.f1_score(Ytrain, pred_tr)
    
    print('F1-score: ',f1_sc)
    print('Accuracy score: ',tr_acc)    


# Ah! metrics are better with just a few lines and a couple more minutes of work. If we ensemble and submit again how do we do on the LB? 0.7305 - good enough for somewhere around 8700. Not bad! But we know that we aren't optimized at all. We've cut some corners on missing values and our random forest is default values. But, what do you know - time for a coffee break soon. Can we do something in 5 minutes that can help us again? Hrmmm... how about a quick grid search?

# In[ ]:


from sklearn.model_selection import GridSearchCV

def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
#GridSearchCV
print('Grid Searching...')
# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "min_samples_split": [2, 3, 5, 7, 10, 12, 15],
              "min_samples_leaf": [2, 3, 5, 7, 10, 12, 15],
              "bootstrap": [False, True],
              "criterion": ["gini", "entropy"]}

clf = RandomForestClassifier(n_estimators=10)

# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid)
grid_search.fit(Xtrain, Ytrain)
report(grid_search.cv_results_)


# Wow - that fast... did manage to grab that coffee. (In reality I ran it for n_estimators=100 not 10). Let's run it with our top ranking search values then.

# In[ ]:


def random_forest_classifier_gs(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(max_depth = None, min_samples_leaf= 3, bootstrap = False, min_samples_split = 7, criterion = 'entropy')
    clf.fit(features, target)
    return clf 

for train_index, cv_index in kf.split(Xtrain, Ytrain):
    Xtrain_K, X_cv = Xtrain[train_index], Xtrain[cv_index]
    Ytrain_K, Y_cv = Ytrain[train_index], Ytrain[cv_index]      
    
    trained_model = random_forest_classifier_gs(Xtrain_K, Ytrain_K)
    

    pred_tr = trained_model.predict(Xtrain.reshape(len(Ytrain),-1))
       
    pred_test = trained_model.predict(Xtest)


# Submitting my resulting file -> 0.76076 which is about 7500 on the LB. Only 15 minutes of work and you grabbed a coffee during the gridsearch.

# Below is the code for creating a submission file

# In[ ]:


#ensemble - majority vote - I just used the default dataframe.mode() function works close enough for a binary choice
#fields = pred.iloc[:, 1:]
#pred['Survived'] = fields.mode(axis=1)
#pred[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
#print(pred.head(10))


# ## Okay, so 15 mins gets me there... where to next?

# Well, that was fun and easy. No sweat. But there are so many holes to plug and places to move up the LB (or kudos from your boss). How much more time do you have? Here are some more quick thoughts on improvements though there are extensive discussions around feature engineering and other things, you need to pick and choose the things you have time for and what should improve your metrics the most. 15 min and in the top 80% not great but still, only 15 mins! Imagine what 15 more would do...

# In no specific order:
# 
# (1) Better engineer our missing values. We could use some sort of predictor to train a model to predict the missing ages. You would need to be careful when doing this in the test set though.
# 
# (2) Try and use all the given data - engineer our own features too. Some people suggest titles, how rich they are, where they embarked, and so on.
# 
# (3) Try different ML methods - RF is very simple, try more complex methods. Be wary of what you do - RF is extremely forgiving. Other methods will need much more data pre-processing to remove ordinal numbers, categorization, and feature scaling.
# 
# (4) Find feature importance
# 
# (5) Stacking and ensembling (extremely important!)
# 
# 
# These discussions highlight some intersting ideas you can try:
# 
# https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline by Poonam Ligade
# 
# https://www.kaggle.com/sinakhorami/titanic-best-working-classifier by Sina
# 
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python by Anisotropic
# 
# https://www.kaggle.com/dongxu027/explore-stacking-lb-0-1463 by DSEverything
# 
# https://mlwave.com/kaggle-ensembling-guide/ - learn this stuff!
# 
# There are plenty of other resources and this is by no means meant to be an exhaustive list.
# 
# 15 minutes won't win you a Kaggle competition, but these basics can at least get you started!
# 
# Cheers
