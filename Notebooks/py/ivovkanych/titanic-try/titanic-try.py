#!/usr/bin/env python
# coding: utf-8

# just my first try

# In[ ]:


#Python 3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model, cross_validation #statistics and ML


# In[ ]:


#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )


#Print to standard output, and see the results in the "log" section below after running your script
print("\n\nTop of the training data:")
print(train.head())

print("\n\nSummary statistics of training data")
print(train.describe())

print("\n\nUnique values for some of the columns")
print(train['Sex'].unique())


# **Filling NAs**

# In[ ]:


train['Age']=train['Age'].fillna(train['Age'].median())
train['Embarked']=train['Embarked'].fillna('S')


# **Formating categorical variables into numeric format**

# In[ ]:


train.loc[train['Sex']=='male','Sex']=0
train.loc[train['Sex']=='female','Sex']=1
train.loc[train['Embarked']=='S','Embarked']=0
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2


# **Linear regression**

# In[ ]:


factors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
linreg=linear_model.LinearRegression()
kf=cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)

predictions = []

for k_train,k_test in kf:
    train_predictors=train[factors].iloc[k_train,:]
    train_target=train["Survived"].iloc[k_train]
    linreg.fit(train_predictors,train_target)
    test_predictions = linreg.predict(train[factors].iloc[k_test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

predictions[predictions>0.5]=1
predictions[predictions<=0.5]=0

accuracy=sum(train['Survived']==predictions)/len(train['Survived'])
print('Linear regression accuracy:',accuracy)


# **Logistic regression**

# In[ ]:


logalg=linear_model.LogisticRegression(random_state=1)
scores=cross_validation.cross_val_score(logalg,train[factors],train["Survived"],cv=3)
print('Logistinc regression accuracy:', scores.mean())


# In[ ]:


test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
test['Age']=test['Age'].fillna(train['Age'].median())
test['Fare']=test['Fare'].fillna(test['Fare'].median())
test['Embarked']=test['Embarked'].fillna('S')

test.loc[test['Sex']=='male','Sex']=0
test.loc[test['Sex']=='female','Sex']=1
test.loc[test['Embarked']=='S','Embarked']=0
test.loc[test['Embarked']=='C','Embarked']=1
test.loc[test['Embarked']=='Q','Embarked']=2

print(test)


# In[ ]:


logalg=linear_model.LogisticRegression(random_state=1)
logalg.fit(train[factors],train['Survived'])
predictions=logalg.predict(test[factors])
submission = pd.DataFrame({"PassengerId":test["PassengerId"],
                          "Survived":predictions})
print(submission)


# **Saving the results**

# In[ ]:


submission.to_csv('voi_titanic_submission1.csv', index=False)

