#!/usr/bin/env python
# coding: utf-8

# # Binary Classification (beginner) - RF x SVM
# Hi. This is my simple solution for the competition Titanic: Machine Learning from Disaster, a binary classification problem.
# I'm newbie. Please, don't hesitate in sending me suggestions.
# Also, I'm not a native english speaker and you will notice that : ) 
# 
# 
# ### Steps:
# 
# 
# 1. Import Libraries and Files
# 2. Preprocessing: dealing with missing values
# 3.  Prepare data (get dummies)
# 4. Nested cross validation to choose the model: Random Forest or SVM  ?
# 5. Submission
# 

# ## 1. Import Libraries and Files

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## 2. Preprocessing: dealing with missing values
# Here I'm doing this separately on train and test set to avoid data leakage.

# In[ ]:


train.isnull().sum()[train.isnull().sum()>0]


# In[ ]:


test.isnull().sum()[test.isnull().sum()>0]


# ### 2.1 - To fill missing values at "Age" : function that considers the person's title to calculate the mean age

# In[ ]:


train['title']=train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
test['title']=test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())


# In[ ]:


newtitles={
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"}


# In[ ]:


train.title=train.title.map(newtitles)
test.title=test.title.map(newtitles)


# In[ ]:


train.groupby(['title','Sex']).Age.mean()


# In[ ]:


def newage (cols):
    title=cols[0]
    Sex=cols[1]
    Age=cols[2]
    
    if pd.isnull(Age):
        if title=='Master' and Sex=='male':
            return 4.57      
        elif title=='Miss' and Sex=='female':
            return 21.8        
        elif title=='Mr' and Sex=='male':
            return 32.37    
        elif title=='Mrs' and Sex=='female':
            return 35.72        
        elif title=='Officer' and Sex=='female':
            return 49
        elif title=='Officer' and Sex=='male':
            return 46.56        
        elif title=='Royalty' and Sex=='female':
            return 40.50        
        else:
            return 43.33
    
    else:
        return Age


# In[ ]:


train['Age']=train[['title', 'Sex','Age']].apply(newage, axis=1)
test['Age']=test[['title', 'Sex','Age']].apply(newage, axis=1)


# ### 2.2 To fill missing values at "Cabin" : "Unknow"

# In[ ]:


train.Cabin=train.Cabin.fillna('Unknow')
test.Cabin=test.Cabin.fillna('Unknow')


# ### 2.3 To fill missing values at "Fare" : mean 

# In[ ]:


test['Fare']=test['Fare'].fillna(test['Fare'].mean())


# ### 2.4 To fill missing values at "Embarked"  : the most comum

# In[ ]:


train.Embarked=train.Embarked.fillna(train.Embarked.value_counts().index[0])


# ## 3 - Prepare data (get dummies)
# Many machine learning algorithms require numerical inputs, so its important transform categoricals variables into numerics. 
# As our categorics values does not seems to have a order, I have used One Hot Encoder.

# In[ ]:


[col for col in train.select_dtypes(include=['object'])]
#Name an Ticket wont be used on model


# In[ ]:


len_train = train.shape[0] 
titanic = pd.concat([train, test], sort=False)


# In[ ]:


titanic['Sex']=pd.get_dummies(titanic['Sex'], drop_first=True)
cabin_dummies=pd.get_dummies(titanic['Cabin'],prefix='Cabin')
embarked_dummies=pd.get_dummies(titanic['Embarked'],prefix='Embarked')
title_dummies=pd.get_dummies(titanic['title'],prefix='title')


# In[ ]:


titanic=pd.concat([titanic,cabin_dummies, embarked_dummies, title_dummies], axis=1)


# In[ ]:


train = titanic[:len_train]
test = titanic[len_train:]


# In[ ]:


train['Survived']=train['Survived'].astype('int')


# ## 4 - Cross validation to choose de model: Random Forest or SVM  ?
# 
# Here, I have used nested cross validation to choose the model, and than non nested to choose best params

# In[ ]:


xtrain = train.drop(['PassengerId', 'Survived', 'Name','Ticket','Cabin','Embarked','title'], axis=1)
ytrain = train['Survived']
xtest = test.drop(['PassengerId', 'Survived', 'Name','Ticket','Cabin','Embarked','title'], axis=1)


# ### 4.1 RandomForest

# In[ ]:


modelrf=RandomForestClassifier(random_state = 1)
paramrf={'n_estimators':[80,100,120], 'max_depth':[4,5,6] }
GSRF=GridSearchCV(estimator=modelrf, param_grid=paramrf, scoring='accuracy', cv=2)
scores=cross_val_score(GSRF, xtrain, ytrain, scoring='accuracy', cv=5)
print(np.mean(scores))


# ### 4.2 SVM

# In[ ]:


pipesvm = make_pipeline(StandardScaler(),SVC(random_state = 1))
param_range = [0.1, 1.0, 10.0]
paramsvm = [{'svc__C': param_range,'svc__kernel': ['linear']}, {'svc__C': param_range,'svc__gamma': param_range,'svc__kernel': ['rbf']}]
GSSVM=GridSearchCV(estimator=pipesvm, param_grid=paramsvm, scoring='accuracy', cv=2)
scores=cross_val_score(GSSVM, xtrain.astype(float), ytrain, scoring='accuracy', cv=5)
print(np.mean(scores))


# ## 5. Submission
# I have chosen Random Forest since his accuracy was slightly better.

# In[ ]:


GSRF.fit(xtrain, ytrain)
print(GSRF.best_params_)


# In[ ]:


model=RandomForestClassifier(n_estimators=120, max_depth=6,random_state = 1)


# In[ ]:


model.fit(xtrain, ytrain)


# In[ ]:


pred=model.predict(xtest)


# In[ ]:


output = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Survived': pred})


# In[ ]:


output.to_csv('submission.csv', index=False)


# In[ ]:


output.head()

