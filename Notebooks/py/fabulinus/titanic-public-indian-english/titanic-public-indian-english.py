#!/usr/bin/env python
# coding: utf-8

# A very simple approach towards 'Titanic' problem. 
# 
# Synopsis: We are asked to predict the survival of passengers on board based on the train data available with us. 
# 
# Problem : There are certain challenges with the train and test dta available, the data has lot of missing,inf values. Ways to over come this issue is detailed below. 
# 
# Survived passengers = 1 - kate winslet 
# Non-Survived passengers = 0 - oh boy ! leonardo dicaprio
# 
# The result columns that we are asked to predict needs to be a binary column, as mentioned above. 
# 
# * Step 1 : Import libraries 
# * Step 2: EDA
# * Step 3: Find columns with missing values and Nan values
# * Step 4: Handle Nan Values using imputer
# * Step 5: Create categorical variables of the columns with values that can be categorized , say Male - Female under Sex , Embarked (port from where the passenger boarded titanic ) can be categorizd into 3.
# * Step 6: Plot graphs to further visualize the data.
# * Step 7: Slpit the data frame into validation and train set data
# * Step 8: Build a model (linear regression) , Fit and predict 
# * Step 9: Score your model

# In[330]:


#Import libraries


# In[331]:


import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[332]:


#Load DATA - Train and Test


# In[333]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[334]:


#EDA


# In[335]:


print('The shape of train data:',df_train.shape)
print('The shape of test data:',df_test.shape)


# In[336]:


df_train.columns


# In[337]:


df_test.columns


# In[338]:


df_train.describe


# In[339]:


df_train.info()


# In[340]:


df_test.info()


# In[341]:


df_train.head(30)


# **Grouping by multiple columns:**
# 
# Name: survived, dtype: int64
# embarked  pclass
# C         1         141
#           2          28
#           3         101
# Q         1           3
#           2           7
#           3         113
# S         1         177
#           2         242
#           3         495
# Na****me: survived, dtype: int64
# 
# Bascially , we are grouping the survived passenger's count based on the place from where they boarded the infamous Titanic
# 

# In[342]:


by_class = df_train.groupby(['Pclass','Embarked'])
count_Survived = by_class['Survived'].count()
print(count_Survived)


# In[343]:


by_age = df_train.groupby(['Age'])
ccount_survived_age = by_age['Survived'].count()
print(ccount_survived_age.head(2))


# 

# based on the .info observation, we see that there are columns with missing values :
# age,cabin,embarked in df_train
# age , cabin in df_test

# In[344]:


#EDA - Graphs


# In[345]:


df_train.Survived.value_counts().plot(kind='bar',alpha=0.9)
plt.title('survived = 1 ')


# In[346]:


plt.scatter(range(df_train.shape[0]),np.sort(df_train.Age),alpha=0.4)
plt.title('age distribution')


# In[347]:


df_train.Pclass.value_counts().plot(kind='barh',alpha=0.5)
plt.title('Pclass')


# In[348]:


df_train.Embarked.value_counts().plot(kind='bar',alpha=0.5)
plt.title('Embarked')


# In[349]:


plt.scatter(range(df_train.shape[0]),np.sort(df_train.Fare),alpha=0.6)
plt.title('Fare distribution')


# In[350]:



train_male = df_train.Survived[df_train.Sex == 'male'].value_counts()

train_female = df_train.Survived[df_train.Sex == 'female'].value_counts()

ind = np.arange(2)
width = 0.3
fig, ax = plt.subplots()
male = ax.bar(ind, np.array(train_male), width, color='r')
female = ax.bar(ind+width, np.array(train_female), width, color='b')
ax.set_ylabel('Count')
ax.set_title('DV count by Gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('DV=0', 'DV=1'))
ax.legend((male[0], female[0]), ('Male', 'Female'))
plt.show()


# Lets build a model and predict the columns

# In[351]:


df_train.columns


# In[352]:


from sklearn.preprocessing import Imputer


# In[353]:


train_cats(df_train)


# In[354]:


df_train.Sex.cat.categories
df_train.Sex = df_train.Sex.cat.codes


# In[355]:


df_train.Embarked.cat.categories
df_train.Embarked = df_train.Embarked.cat.codes


# In[356]:


by_sex_class = df_train.groupby(['Sex','Pclass'])
def impute_median(series):
    return series.fillna(series.mean())
df_train.Age = by_sex_class.Age.transform(impute_median)
df_train.Age.tail()


# In[367]:


X_train = df_train[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]
y_train = df_train['Survived']
X_test = df_test[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]


# ************Slpit the data frame into validation and train set data

# In[382]:


X_trn,X_val,y_trn,y_val = train_test_split(X_train,y_train,test_size=0.469,random_state=42)
m = lm.LogisticRegression()
m.fit(X_trn,y_trn)
pred = m.predict(X_val)
pred


# In[385]:


#my_submission = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': pred})
#my_submission.to_csv('submission_rakesh.csv', index=False)

submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": pred
    })
submission


# In[386]:


print(submission.to_string())


# In[384]:


from sklearn.metrics import accuracy_score

accuracy_score(y_val,pred)


# In[392]:


X_train = df_train[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]
y_train = df_train['Survived']
X_test = df_test[['Pclass','Fare','Parch','Sex','Embarked','SibSp','Age']]


# In[394]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_trn,y_trn)
Y_pred = random_forest.predict(X_val)
Y_pred


# In[398]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Y_pred
    })
print(submission.to_string())


# In[395]:


accuracy_score(y_val,Y_pred)


# In[ ]:




