#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Load Libraries
import pandas as pd
from pandas import Series,DataFrame
import csv
import sklearn
from sklearn.linear_model import LogisticRegression


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')


# Read train dataset

# In[ ]:


df=pd.read_csv("../input/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# The age column has 177 missing values. To handle the missing values lets replace the nan values with median age values

# In[ ]:


df['Age'].fillna(df['Age'].median(), inplace=True)


# In[ ]:


df.describe()


# In[ ]:


sns.barplot(x="Sex", y="Survived", data=df)


# In[ ]:


survived_sex=df[df['Survived']==1]['Sex'].value_counts()
dead_sex=df[df['Survived']==0]['Sex'].value_counts()
df_survived=pd.DataFrame([survived_sex, dead_sex])
df_survived.index=['Survived','Dead']
df_survived.plot(kind='bar',stacked=True)


# Women are more likely to survive than men

# In[ ]:


plt.hist([df[df['Survived']==1]['Age'], df[df['Survived']==0]['Age']], stacked=True, color=['g','r'], bins=30,
         label=['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# In[ ]:


plt.hist([df[df['Survived']==1]['Fare'],df[df['Survived']==0]['Fare']], stacked=True, color=['g','r'],bins=30, label=['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# Passengers with less fare tickets are more likely to die than with expensive tickets
# 
# 

# In[ ]:


ax = plt.subplot()
ax.set_ylabel('Average fare')
df.groupby('Pclass').mean()['Fare'].plot(kind='bar',ax = ax)


# In[ ]:


sns.factorplot('Sex', kind='count', data=df)


# In[ ]:


sns.factorplot('Pclass',kind='count',data=df, hue='Sex')


# Proportion of passengers survived based on their passenger class

# In[ ]:


xt=pd.crosstab(df['Pclass'],df['Survived'])
xt


# In[ ]:


xt.plot(kind='bar',stacked=True, title='Survival Rate by Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')


# Random Forest for training dataset

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing


# Create the decision trees

# In[ ]:


np.random.seed(12)


# In[ ]:


label_encoder=preprocessing.LabelEncoder()


# In[ ]:


# Convert sex and embarked variables to numeric
df['Sex']=label_encoder.fit_transform(df['Sex'].astype('str'))
df['Embarked']=label_encoder.fit_transform(df['Embarked'].astype('str'))


# In[ ]:


# Initialize the model
rf_model=RandomForestClassifier(n_estimators=1000, max_features=2,oob_score=True)
features=['Sex','Pclass','Embarked','Age','Fare']


# In[ ]:


# Train the model
rf_model.fit(X=df[features],y=df['Survived'].astype('str'))
print("OOB accuracy: ")
print(rf_model.oob_score_)


# Features with higher importance were more influential in creating the model, indicating a stronger association with the response variable.
# 

# Feature importance for our random forest model:

# In[ ]:


for feature, imp in zip(features,rf_model.feature_importances_):
    print(feature,imp)


# Use random forest model to make predictions on the test dataset

# Read test dataset

# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


test.describe()


# Impute the median age for NA age values

# In[ ]:


test['Age'].fillna(test['Age'].median(), inplace=True)


# In[ ]:


test.describe()


# In[ ]:


# Convert sex and embarked variables of test dataset to numeric
test['Sex']=label_encoder.fit_transform(test['Sex'].astype('str'))
test['Embarked']=label_encoder.fit_transform(test['Embarked'].astype('str'))


# In[ ]:


test.head()


# In[ ]:


test.fillna(test.mean(), inplace=True)


# In[ ]:


# Predictions for test set
test_preds = rf_model.predict(X=test[features])


# In[ ]:


submission=pd.DataFrame({"PassengerId": test["PassengerId"], "Survived":test_preds})
submission.to_csv('titanic1.csv', index=False)

