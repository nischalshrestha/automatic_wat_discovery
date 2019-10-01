#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

get_ipython().magic(u'matplotlib inline')


# In[ ]:





# ### Read the data and save to a dataframe

# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:





# ### Quick look at the data

# In[ ]:


data.head(10)


# In[ ]:


data.tail(10)


# In[ ]:





# In[ ]:


sns.pairplot(data,x_vars=['Age','Fare','Pclass'],y_vars='Survived',kind='reg',size=7)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:





# ### Check for any Null values 

# In[ ]:


data.isna().sum()


# In[ ]:





# ### Further look into the data
# 
# Look into the Age column, Cabin Column and the embarked columns. 

# #### Age

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data.groupby('Age').Survived.value_counts(dropna=False)


# In[ ]:


data.Age.describe()


# In[ ]:


data.Age.agg(['min','max','mean','std'])


# In[ ]:


data.Age.agg(['min','max','mean','std']).plot(kind = 'barh')


# In[ ]:





# In[ ]:


age_survival = data.loc[data.Survived == 1,'Age'].value_counts().sort_index().plot(figsize=(13,8))

age_survival.set_xlabel('Age')
age_survival.set_ylabel('Survival')


# In[ ]:





# In[ ]:


data.loc[(data['Survived']==1) & (data['Sex']=='female') & (data['Age'])]


# In[ ]:


sns.boxplot(x =data.Sex =='female',y=data['Survived'])


# In[ ]:


data.loc[data.Sex=='female','Survived'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:


data.loc[(data['Survived']==1) & (data['Sex']=='female') & (data['Age'])].mean()


# In[ ]:





# In[ ]:


sns.pairplot(data,x_vars='Age',y_vars='Survived',kind='reg',size=10)


# In[ ]:





# #### Embarked

# In[ ]:





# In[ ]:


pd.crosstab(data.Survived,data.Embarked).plot(kind = 'bar')


# In[ ]:





# In[ ]:


data.Embarked.value_counts(dropna=False)


# In[ ]:


data['Embarked'] = data.Embarked.map({'S':0,'C':1,'Q':2})


# In[ ]:


data.Embarked.value_counts()


# In[ ]:


data['Embarked'] = data.Embarked.fillna(value = 0.0)


# In[ ]:





# In[ ]:


data.Embarked.value_counts(dropna=False)


# In[ ]:


data.Embarked.shape


# In[ ]:





# In[ ]:


data.head()


# In[ ]:


data['Embarked'].head()


# In[ ]:


data.Embarked.shape


# In[ ]:


data.Embarked.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:


sns.pairplot(data,x_vars='Embarked',y_vars='Survived',kind='reg',size=10)


# In[ ]:





# In[ ]:





# #### Cabin

# In[ ]:


data.Cabin.value_counts().head()


# In[ ]:





# In[ ]:


data[(data.Survived ==1) & (data.Cabin)]


# In[ ]:





# I have decided to Drop the Cabin Column because it has too many missing values 

# In[ ]:





# In[ ]:





# ### Categorizing the Age column and create dummy variables. 

# In[ ]:


data.loc[data['Age'] <= 15,'Age'] = 0 

data.loc[(data['Age'] > 15) & (data['Age'] <= 30), 'Age'] = 1

data.loc[(data['Age'] > 30) & (data['Age'] <= 50), 'Age'] = 2

data.loc[(data['Age'] > 50),'Age'] = 3


# In[ ]:





# In[ ]:


data.Age.head()


# In[ ]:





# In[ ]:


data.Age.isna().sum()


# In[ ]:





# In[ ]:


data.dropna(subset=['Age'],axis ='index',how='all',inplace=True)


# In[ ]:


data.Age.isna().sum()


# In[ ]:


data.Age.value_counts(dropna=False)


# In[ ]:


data.Age.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:


sns.pairplot(data,x_vars='Age',y_vars='Survived',size=10,kind='reg')


# In[ ]:





# In[ ]:


data.head(10)


# In[ ]:


data.Age.isna().sum()


# In[ ]:





# In[ ]:





# ### Create Dummy Variables for Sex Column

# In[ ]:





# In[ ]:


pd.crosstab(data.Survived,data.Sex).plot(kind='bar')


# In[ ]:





# In[ ]:





# In[ ]:


data['Sex'] = data.Sex.map({'female':0,'male':1})


# In[ ]:


data.Sex.values


# In[ ]:





# In[ ]:


data.head(10)


# In[ ]:





# In[ ]:


data.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:


data.head(100)


# ### Create a model using Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns


# In[ ]:


feature_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']


# In[ ]:


sns.pairplot(data,x_vars=feature_cols,y_vars = 'Survived',kind = 'reg',size = 4,aspect=0.9)


# In[ ]:





# In[ ]:





# In[ ]:


feature_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']

X = data[feature_cols]

y = data.Survived


# In[ ]:


print(type(X))
print(X.shape)


# In[ ]:


print(type(y))
print(y.shape)


# In[ ]:





# In[ ]:


#Instantiate the Model

linreg = LinearRegression()


# In[ ]:


# Fit the model 

linreg.fit(X,y)


# In[ ]:





# In[ ]:


linreg.intercept_


# In[ ]:


linreg.coef_


# In[ ]:


feature_list = list(zip(feature_cols,linreg.coef_))


# In[ ]:





# In[ ]:


feature_list


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### check the cross validation score. 

# In[ ]:


# 10 fold cross validation with all 4 features

linreg = LinearRegression()

score = cross_val_score(linreg,X,y,cv=10,scoring='neg_mean_squared_error')


# In[ ]:


score


# In[ ]:





# In[ ]:


# Make the scores +

msc_sc = -score
print(msc_sc)


# In[ ]:





# In[ ]:


# Calculate te RMSE

rmse = np.sqrt(msc_sc)
print(rmse)


# In[ ]:





# In[ ]:


## Print the mean of RMSE

print(rmse.mean())


# In[ ]:





# In[ ]:





# ### Make Predictions

# In[ ]:


# Load the test set 

test = pd.read_csv('../input/test.csv')


# In[ ]:


test.head()


# In[ ]:





# #### Prepare the data. 

# In[ ]:


test['Sex'] = test.Sex.map({'female':0,'male':1})


# In[ ]:


test.Sex.value_counts()


# In[ ]:





# In[ ]:


test.Embarked.value_counts()


# In[ ]:


test['Embarked'] = test.Embarked.map({'S':0,'C':1,'Q':2})


# In[ ]:


test.Embarked.value_counts()


# In[ ]:





# In[ ]:


test.loc[test['Age'] <= 15,'Age'] = 0 

test.loc[(test['Age'] > 15) & (test['Age'] <= 30), 'Age'] = 1

test.loc[(test['Age'] > 30) & (test['Age'] <= 50), 'Age'] = 2

test.loc[(test['Age'] > 50),'Age'] = 3


# In[ ]:





# In[ ]:


test.isna().sum()


# In[ ]:


test.Age.dropna(axis='index',how='any',inplace=True)


# In[ ]:


test['Age'] = test.Age.isna().sum()


# In[ ]:





# In[ ]:


test.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# ### Select X

# In[ ]:


feature_cols = ['Pclass','Sex','Age','Embarked','SibSp']

X_test = test[feature_cols]


# In[ ]:


linreg = LinearRegression()

linreg.fit(X,y)


# In[ ]:





# In[ ]:


y_pred = linreg.predict(X)


# In[ ]:


y_pred[X_test]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Fit and test a classification model. 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


feature_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']

X = data[feature_cols]

y = data.Survived


# In[ ]:


logreg = LogisticRegression()

logreg.fit(X,y)


# In[ ]:





# ### Check the cross Validation score

# In[ ]:


score = cross_val_score(logreg,X,y,cv=10,scoring='neg_mean_squared_error')

score


# In[ ]:





# In[ ]:


mse = - score


# In[ ]:


## NOw calculate the RMSE

rmse = np.sqrt(mse)
print(rmse)


# In[ ]:





# In[ ]:





# ### Predict from the loaded test set

# In[ ]:


test_cols = ['Pclass','Sex','Age','Embarked','SibSp','Fare']

X_test = test[test_cols]


# In[ ]:


X


# In[ ]:


X_test


# In[ ]:


X_test.isna().sum()


# In[ ]:


X_test.dtypes


# In[ ]:





# In[ ]:


X_test.Fare.fillna(X_test.Fare.mean(),inplace=True)


# In[ ]:





# In[ ]:


X_test.isna().sum()


# In[ ]:





# In[ ]:





# In[ ]:


logreg = LogisticRegression()

logreg.fit(X,y)


# In[ ]:


y_pred = logreg.predict(X_test)


# In[ ]:


y_pred


# In[ ]:





# In[ ]:


pd.get_option('display.max_rows')


# In[ ]:


pd.set_option('display.max_rows',None)


# In[ ]:





# In[ ]:


X_test.shape


# In[ ]:


test.PassengerId.shape


# In[ ]:





# ### Create the kaggle Submission file

# In[ ]:


# Create a pandas Dataframe

pd.DataFrame({'PasssngerId':test.PassengerId,'Survived':y_pred})


# In[ ]:





# In[ ]:


# now save PassengerId columns as the index

pd.DataFrame({'PasssngerId':test.PassengerId,'Survived':y_pred}).set_index('PasssngerId')


# In[ ]:





# In[ ]:





# In[ ]:


# Finally Convert the file to a CSV file 

pd.DataFrame({'PassengerId':test.PassengerId,'Survived':y_pred}).set_index('PassengerId').to_csv('Titaanic log reg2.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




