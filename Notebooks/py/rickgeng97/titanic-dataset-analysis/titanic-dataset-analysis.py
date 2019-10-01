#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Wrangling
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#training and testing
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# **First we check datasets dimension and information**

# In[ ]:


#import dataset
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')


# In[ ]:


#check column names
train_raw.columns.values


# In[ ]:


#dataset dimension
train_raw.shape


# In[ ]:


#datasets information
print(train_raw.info())
print('_'*40)
print(test_raw.info())


# **Clean Datasets**
# 1. Fill in Missing Values
# 2. Remove Outliers

# In[ ]:


#check Missing Values
print(train_raw.isnull().sum())
print('_'*40)
print(test_raw.isnull().sum())


# In[ ]:


#Fill in missing values expect cabin column 
train_raw['Age'] = train_raw.Age.fillna(train_raw.Age.mean())
test_raw['Age'] = test_raw.Age.fillna(test_raw.Age.mean())
train_raw['Embarked'] = train_raw.Embarked.fillna(train_raw.Embarked.mode()[0])
test_raw['Embarked'] = test_raw.Embarked.fillna(test_raw.Embarked.mode()[0])
test_raw['Fare'] = test_raw.Fare.fillna(test_raw.Fare.mean())
print(train_raw.isnull().sum())
print('_'*40)
print(test_raw.isnull().sum())


# In[ ]:


#Check outliers for continuous variables 
plt.figure(figsize = (15,5))
plt.subplot(2,1,1)
sns.boxplot(x = 'Age', data = train_raw)
plt.subplot(2,1,2)
sns.boxplot(x = 'Fare', data = train_raw)


# Now we analyse distributions of data  

# In[ ]:


#Age Analysis
plt.figure(figsize = (15,10))
plt.subplot(2,1,1)
ax1 = sns.distplot(train_raw.Age)
ax1.set(xlabel = "Age", ylabel = "Percentage")
plt.subplot(2,1,2)
ax2 = sns.boxplot(train_raw.Age)
for i in np.arange(0,1.25,0.25):
    print("{0:.0f}%:".format(i * 100),train_raw['Age'].quantile(i))


# In[ ]:


#Fare Analysis
plt.figure(figsize = (15,10))
plt.subplot(2,1,1)
ax1 = sns.distplot(train_raw.Fare)
ax1.set(xlabel = "Fare", ylabel = "Percentage")
plt.subplot(2,1,2)
ax2 = sns.boxplot(train_raw.Fare)
for i in np.arange(0,1.25,0.25):
    print("{0:.0f}%:".format(i * 100), train_raw['Fare'].quantile(i))


# **Survival Rate Analysis**

# In[ ]:


train = train_raw.copy()


# In[ ]:


#Total Survival Rate
train.Survived.mean()


# In[ ]:


#Sex classes and survival rate for each class
print(train.groupby('Sex').size())
print('_'*40)
print(train.groupby('Sex').mean()['Survived'].sort_values(ascending = False))


# In[ ]:


#Partition Age
age_partition_size = pd.DataFrame()
age_partition = pd.DataFrame(train[['Survived','Age']])
age_partition['Partition'] = pd.cut(train['Age'], [0,10,20,30,40,50,60,70,80])

age_partition_size['Survived'] = age_partition.groupby('Partition').mean()['Survived']
age_partition_size['Size'] = age_partition.groupby('Partition').size()
age_partition_size


# In[ ]:


#Embarked classes and survival rate for each class
print(train.groupby('Embarked').size())
print('_'*40)
print(train.groupby('Embarked').mean()['Survived'].sort_values(ascending = False))


# In[ ]:


#SibSp classes and survival rate for each class
print(train.groupby('SibSp').size())
print('_'*40)
print(train.groupby('SibSp').mean()['Survived'].sort_values(ascending = False))


# In[ ]:


#Parch classes and survival rate for each class
print(train.groupby('Parch').size())
print('_'*40)
print(train.groupby('Parch').mean()['Survived'].sort_values(ascending = False))


# **Analysing Correlations**

# In[ ]:


#correlation
sns.pairplot(train)


# **Classifing and Testing using ML models **

# In[ ]:


#Drop columns
train_Predict = train.copy()
label = train_Predict.Survived
train_Predict = train_Predict.drop(['PassengerId','Survived','Name','Ticket','Cabin'], axis = 1)


# In[ ]:


#One hot encoding 
def sex_binary(sex):
    if sex == "male": 
        return 0
    else:
        return 1
    
def embarked_class(embarked):
    if embarked == 'S':
        return 0
    elif embarked == 'C':
        return 1
    elif embarked == 'Q':
        return 2


# In[ ]:


train_Predict['Age'] = age_partition['Partition']
train_Predict['Sex'] = train['Sex'].apply(sex_binary)
train_Predict['Embarked'] = train['Embarked'].apply(embarked_class)
train_Predict['Age'] = age_partition['Partition'].apply(lambda x: x.mid)


# In[ ]:


#training set 
train_Predict.tail()


# In[ ]:


#Decision Tree Classifier
clf = DecisionTreeClassifier()
x_train, y_train, x_test, y_test = train_test_split(train_Predict, label, train_size = 0.8, random_state = 0)
clf.fit(x_train, x_test)


# In[ ]:


#Moment of Truth
y_pred = clf.predict(y_train)
accuracy_score(y_test, y_pred)


# In[ ]:


#Random Forest Classifier
clf = RandomForestClassifier()
x_train, y_train, x_test, y_test = train_test_split(train_Predict, label, train_size = 0.8, random_state = 0)
clf.fit(x_train, x_test)


# In[ ]:


#Moment of Truth
y_pred = clf.predict(y_train)
accuracy_score(y_test, y_pred)


# 
