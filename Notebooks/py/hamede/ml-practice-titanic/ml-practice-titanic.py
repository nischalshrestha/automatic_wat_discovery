#!/usr/bin/env python
# coding: utf-8

# # 1. Import Libraries

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


# visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().magic(u'matplotlib inline')
pd.options.mode.chained_assignment = None  # default='warn'

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# ALl fastai imports
# from fastai.imports import * # it imports has basic packages like pandas, numpy
# from fastai.structured import *
# from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *


# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# # 2. Load datasets

# In[ ]:


import os
os.listdir('../input')


# In[ ]:


raw_train_df = pd.read_csv("../input/train.csv")
raw_test_df = pd.read_csv("../input/test.csv")



print(raw_train_df.columns.values)


# In[ ]:


# show preview from data
raw_train_df.head()


# In[ ]:


raw_train_df.tail()


# In[ ]:


# train_df.info()
# print('_'*40)
# test_df.info()


# In[ ]:


raw_train_df.describe()


# In[ ]:


raw_test_df.head()


# # 2. Exploratory Data Analysis
# 
# ## Missing values (NaNs)

# In[ ]:


#missing values in training set
train_missing_data = raw_train_df.isnull()
#missing values in test set
test_missing_data = raw_test_df.isnull()


# In[ ]:


train_missing_data.head(5)


# In[ ]:


#function to print the count of missing data
def print_missing_count(df):
    for column in df.columns.values.tolist():
        print(column)
        print(df[column].value_counts())
        print("")


# In[ ]:


print_missing_count(train_missing_data)


# Missing values found in training data
# * Age: 177 counts
# * Cabin: 204 counts
# * Embarked: 2 counts

# In[ ]:


#drop rows with no 'Embarked' information
raw_train_df = raw_train_df.dropna(axis=0,subset=['Embarked'])


# In[ ]:


#mean age in training and testing set
mean_age = (raw_train_df['Age'].mean() + raw_test_df['Age'].mean())/2
mean_age
#replace NaNs in Age column with the mean
raw_train_df['Age'].replace(np.nan, mean_age, inplace=True)


# In[ ]:


#check if there are any more NaNs
print(raw_train_df['Age'].isnull().value_counts())
print(raw_train_df['Embarked'].isnull().value_counts())


# In[ ]:


#print_missing_count(test_df.isnull())


# In[ ]:


#mean fare in training and testing set
mean_fare = (raw_train_df['Fare'].mean() + raw_test_df['Fare'].mean())/2
mean_fare
#replace NaNs in fare column with the mean
raw_test_df['Fare'].replace(np.nan, mean_fare, inplace=True)
#replace NaNs in Age column with the mean
raw_test_df['Age'].replace(np.nan, mean_age, inplace=True)


# In[ ]:


#check if there are any more NaNs
print(raw_test_df['Fare'].isnull().value_counts())
print(raw_test_df['Age'].isnull().value_counts())


# ## Create Validation Set

# In[ ]:


train_df, valid_df = train_test_split(raw_train_df, test_size=0.2)


# In[ ]:


train_df.shape


# In[ ]:


valid_df.shape


# In[ ]:


#distribution of passenger age
sns.distplot(train_df[['Age']], kde=False)


# In[ ]:


#count of passenger gender
sns.countplot(train_df['Sex'])


# # 3. Random forrest
# 
# An initial model with below selected input features are created
# 1.  Age
# 2. Embarked
# 3. Sex
# 4. Pclass
# 5. SibSp
# 6. Parch
# 7. Fare

# In[ ]:


def encode_df(df):
    df['Embarked']= df.Embarked.astype('category')
    df['Sex']= df.Sex.astype('category')
    df['Embarked_code'] = df.Embarked.cat.codes
    df['Sex_code'] = df.Sex.cat.codes
    return df

def score_model(train_feature, train_label, test_feature, test_label, model):
    print("Train Score: ", model.score(train_feature, train_label), "\nTest Score: ", model.score(test_feature, test_label))

    
feature_set1 = ['Pclass','Sex_code','Age','SibSp','Parch','Fare', 'Embarked_code']
label_name = 'Survived'


# In[ ]:


train_df = encode_df(train_df)
valid_df = encode_df(valid_df)

train_df1 = train_df.copy()
valid_df1 = valid_df.copy()


# In[ ]:


train_df1.head()


# In[ ]:


valid_df1.head()


# In[ ]:


train_df1 = train_df1.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
train_df1.head()


# In[ ]:


m = RandomForestClassifier(n_estimators=10, max_depth=5,  max_features='sqrt', random_state=123)
m.fit(train_df1[feature_set1], train_df1[label_name])

test_df = encode_df(raw_test_df)
test_pred = m.predict(test_df[feature_set1])
test_df['Survived'] = test_pred

score_model(train_df1[feature_set1], train_df1[label_name], valid_df1[feature_set1], valid_df1[label_name], m)


# In[ ]:


# res_df = pd.DataFrame({
#     'PassengerId':test_df['PassengerId'],
#     'Survived':test_df['Survived']
# })



# res_df.to_csv('output.csv', index = False)
# res_df.head()


# In[ ]:


#print_missing_count(test_df.isnull())


# ## Create new feature 'AgeGroup'
# 
# Create a feature age group with three different groups
# - Children
# - Adults
# - Elderly

# In[ ]:


def addAgeGroup(df):
    bins = [0, 12, 19, 65, np.inf]
    labels=[0,1,2,3]
    temp_df = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
    df['AgeGroup'] = temp_df.copy()
    return df


# In[ ]:


train_df2 = addAgeGroup(train_df1)
valid_df2 = addAgeGroup(valid_df1)
test_df = addAgeGroup(test_df)


# In[ ]:


train_df2[['Age', 'AgeGroup']].head()


# In[ ]:


ax = sns.countplot(train_df2['AgeGroup'])
ax.set_xticklabels(['Child', 'Teenager', 'Adult', 'Elderly'])
plt.show()


# ## Train RF using new feature 'AgeGroup'

# In[ ]:


feature_set2 = ['Pclass','Sex_code','AgeGroup','SibSp','Parch','Fare', 'Embarked_code']


# In[ ]:


#train_df2 = encode_df(train_df2)
t = RandomForestClassifier(n_estimators=30, max_depth = 3, max_features='sqrt', random_state = 123)
t.fit(train_df2[feature_set2], train_df2[label_name])
#t.score(train_df2[['Pclass','Sex_code','AgeGroup','SibSp','Parch','Fare', 'Embarked_code']], train_df1['Survived'])

# test_pred2 = m.predict(test_df2[feature_set1])
# test_df2['Survived'] = test_pred2
score_model(train_df2[feature_set2], train_df2[label_name], valid_df2[feature_set2], valid_df2[label_name], t)


# # Feature Importance

# In[ ]:


get_ipython().magic(u'pinfo2 rf_feat_importance')


# In[ ]:


m.feature_importances_


# In[ ]:


train_df1.columns[0:len(m.feature_importances_)]


# In[ ]:


fi = rf_feat_importance(m, train_df1[feature_set2])
fi


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi)


# In[ ]:


feature_set3 = ['Pclass','Sex_code','AgeGroup','Fare']
p = RandomForestClassifier(n_estimators=30, max_depth = 3, max_features='sqrt', random_state = 123)
p.fit(train_df2[feature_set3], train_df2[label_name])
score_model(train_df2[feature_set3], train_df2[label_name], valid_df2[feature_set3], valid_df2[label_name], p)


# In[ ]:


test_pred = p.predict(test_df[feature_set3])
test_df['Survived'] = test_pred

res_df = pd.DataFrame({
    'PassengerId':test_df['PassengerId'],
    'Survived':test_df['Survived']
})



res_df.to_csv('output.csv', index = False)
res_df.head()


# In[ ]:




