#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


# Read the data
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')


# In[ ]:


train.isnull().sum()


# ### Nan "Embarked"  fill

# In[ ]:


train["Embarked"] = train["Embarked"].fillna("S")


# In[ ]:


test.isnull().sum()


# ### Nan "Fare" fill

# In[ ]:


test["Fare"] = test["Fare"].fillna(test["Fare"].median())


# ### Sex and Embarked is repaced at number

# In[ ]:


train['Sex'].replace(['male','female'],[0,1],inplace=True)
train['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

test['Sex'].replace(['male','female'],[0,1],inplace=True)
test['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)

train.head(5)


# # Predict "Age" with a random forest.

# In[ ]:


data_age = pd.concat([train, test])
data_age.shape


# In[ ]:


age_test = data_age[data_age['Age'].isnull()]
age_train = data_age.dropna(subset=['Age'])


# In[ ]:


age_train_y = age_train.Age
predictor_cols = ["Pclass", "Sex", "Fare", 'Embarked']

# Create training predictors data
age_train_X = age_train[predictor_cols]

from sklearn.ensemble import RandomForestRegressor
my_model = RandomForestRegressor()
my_model.fit(age_train_X, age_train_y)


# In[ ]:


age_test_X = age_test[predictor_cols]
predicted_age = my_model.predict(age_test_X)


# In[ ]:


age = pd.DataFrame({'PassengerId': age_test.PassengerId, 'Age': predicted_age})
age.head(5)


# ### Insert the predicted "Age".

# In[ ]:


for i in range(263):
    j = age.iloc[i,1]
    train.loc[(train.PassengerId==j),'Age'] = age.iloc[i,0]


# In[ ]:


for i in range(263):
    j = age.iloc[i,1]
    test.loc[(test.PassengerId==j),'Age'] = age.iloc[i,0]


# # Predict "Survived" with a random forest.

# In[ ]:


train_y = train.Survived
predictor_cols = ["Pclass", "Sex", "Age", "Fare", 'Embarked']

# Create training predictors data
train_X = train[predictor_cols]

from sklearn.ensemble import RandomForestRegressor
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)



# In[ ]:


# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_Survived = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_Survived)


# ### make predicted_Survived an integer.

# In[ ]:


predicted_Survived=np.round(predicted_Survived)
predicted_Survived = list(map(int, predicted_Survived))
print(predicted_Survived)


# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted_Survived})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

