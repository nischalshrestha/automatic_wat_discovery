#!/usr/bin/env python
# coding: utf-8

# 

# In[125]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[3]:


data = pd.read_csv('../input/train.csv')


# In[4]:


data.count()


# In[5]:


data.describe()


# Females survived - 233
# Females not survived -81
# 
# Males survived - 109
# Males not survived - 468
# 
# Percentage of Female survival is **74.2%**. 
# 
# Percentage of Male survival is **18.8%**

# In[6]:


# survived_female_ages_age_NaN = data[(data['Survived'] == 1) & (data['Sex'] == 'female') & (~survived_female_ages['Age'].notna())]
# len(survived_female_ages_age_NaN)

#All females
female_ages = data[(data['Sex'] == 'female')]

#Survived females
# survived_female_ages = data[(data['Survived'] == 1) & (data['Sex'] == 'female')]


# In[7]:


import seaborn as sns


# In[8]:


#Survived female ages and ages with NaN
female_ages_survived_age = female_ages[(female_ages['Survived'] == 1) & (female_ages['Age'].notna())]
print(len(female_ages_survived_age))
female_ages_survived_age_NaN = female_ages[(female_ages['Survived'] == 1) & (~female_ages['Age'].notna())]
print(len(female_ages_survived_age_NaN))


# In[9]:


def drawBoxplot(datacolumen):
    sns.set_style("whitegrid")
    sns.boxplot(x=datacolumen)
    sns.swarmplot(x=datacolumen, color='black')


# In[10]:


drawBoxplot(female_ages_survived_age['Age'])


# In[11]:


female_ages_survived_age['Age'].mean()


# In[12]:


#Not Survived female ages and ages with NaN
female_ages_not_survived_age = female_ages[(female_ages['Survived'] == 0) & (female_ages['Age'].notna())]
print(len(female_ages_not_survived_age))
female_ages_not_survived_age_NaN = female_ages[(female_ages['Survived'] == 0) & (~female_ages['Age'].notna())]
print(len(female_ages_not_survived_age_NaN))


# In[13]:


drawBoxplot(female_ages_not_survived_age['Age'])


# In[14]:


female_ages_not_survived_age['Age'].mean()


# In[15]:


data_replaced_ages = data.copy()


# In[16]:


# replace NaN with survived females means
data_replaced_ages.loc[(data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Sex'] == 'female') & (data_replaced_ages['Age'].isnull()), 'Age'] = female_ages_survived_age['Age'].mean()


# In[17]:


#replace NaN with not survived females
data_replaced_ages.loc[(data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Sex'] == 'female') & (data_replaced_ages['Age'].isnull()), 'Age'] = female_ages_not_survived_age['Age'].mean()


# In[18]:


#draw survived males with age
drawBoxplot(data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Age'].notna())]['Age'])


# In[19]:


#draw not survived males with age
drawBoxplot(data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Age'].notna())]['Age'])


# In[20]:


#replace NaN with survived males
survive_males = data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Age'].notna())]


# In[21]:


data_replaced_ages.loc[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 1) & (data_replaced_ages['Age'].isnull()), 'Age'] = survive_males['Age'].median()


# In[22]:


#replace NaN with Not survived males
not_survived_males = data_replaced_ages[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Age'].notna())]


# In[23]:


data_replaced_ages.loc[(data_replaced_ages['Sex'] == 'male') & (data_replaced_ages['Survived'] == 0) & (data_replaced_ages['Age'].isnull()), 'Age'] = not_survived_males['Age'].median()


# In[24]:


data_replaced_ages.head(10)


# In[25]:


# Analysis of based on Pclass
data_replaced_ages_females = data_replaced_ages[(data_replaced_ages['Sex'] == 'female')]
# survived females with pclass
data_replaced_ages_females.groupby(['Survived','Pclass']).size().plot.bar()


# In[26]:


# Analysis of based on Pclass
data_replaced_ages_males = data_replaced_ages[(data_replaced_ages['Sex'] == 'male')]
pd.crosstab(data_replaced_ages_males['Survived'],data_replaced_ages_males['Pclass']).plot.bar()


# In[27]:


import matplotlib.pyplot as plt


# In[28]:


# check ages of survived and not survived females
plt.subplots(figsize=(15,10))
sns.swarmplot(x='Sex', y='Age', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'Age']], dodge=True).set_title('Survival based on Age')


# In[29]:


# check ages of survived and not survived females
plt.subplots(figsize=(30,8))
sns.swarmplot(x='Sex', y='SibSp', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'SibSp']], dodge=True).set_title('Survival vs Siblings and spouse')


# In[30]:


plt.subplots(figsize=(30,8))
sns.swarmplot(x='Sex', y='Parch', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'Parch']], dodge=True).set_title('Survival vs parents and child')


# In[31]:


plt.subplots(figsize=(15,10))
sns.swarmplot(x='Sex', y='Fare', hue='Survived', data=data_replaced_ages[['Survived','Sex', 'Fare']], dodge=True).set_title('Survival based on Fare')


# In[32]:


# Analysis on Cabins
data_cabins = data_replaced_ages[data_replaced_ages['Cabin'].notna()]
pd.crosstab(data_cabins['Sex'], data_cabins['Survived']).plot.bar()


# In[33]:


data_cabins_null = data_replaced_ages[data_replaced_ages['Cabin'].isnull()]
pd.crosstab(data_cabins_null['Sex'], data_cabins_null['Survived']).plot.bar()


# In[34]:


data_replaced_ages.loc[data_replaced_ages['Cabin'].notna(), 'Cabin_Status'] = 1
data_replaced_ages.loc[data_replaced_ages['Cabin'].isnull(), 'Cabin_Status'] = 0


# In[35]:


data_replaced_ages.head()


# In[36]:


#Analysis on survival based on Embarked
#data_replaced_ages.loc[data_replaced_ages['Embarked'].isnull()]
tmp = data_replaced_ages[(data_replaced_ages['Sex'] == 'male')]
pd.crosstab(tmp['Survived'],tmp['Embarked']).plot.bar().set_title('Male survival analysis with Embarked')

tmp1 = data_replaced_ages[(data_replaced_ages['Sex'] == 'female')]
pd.crosstab(tmp1['Survived'],tmp1['Embarked']).plot.bar().set_title('Female survival analysis with Embarked')


# In[37]:


# data_replaced_ages.loc[data_replaced_ages['Embarked'].isnull()] = 'S'
data_replaced_ages.loc[data_replaced_ages['Embarked'].isnull(), 'Embarked'] = 'S'


# In[38]:


from sklearn.datasets import load_iris


# In[39]:


import sklearn.model_selection as model_selection


# In[40]:


final_data_set = data_replaced_ages.drop(['Name', 'Ticket', 'Cabin'], 1)


# In[44]:


test_dataset = pd.read_csv('../input/test.csv')


# In[45]:


test_dataset.count()


# In[49]:


test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Sex'] == 'male'), 'Age'] = (not_survived_males['Age'].median() + survive_males['Age'].median())/2


# In[52]:


test_dataset.loc[(test_dataset['Age'].isnull()) & (test_dataset['Sex'] == 'female'), 'Age'] = (female_ages_survived_age['Age'].mean() + female_ages_not_survived_age['Age'].mean())/2


# In[56]:


test_dataset.loc[test_dataset['Cabin'].notna(), 'Cabin_Status'] = 1
test_dataset.loc[test_dataset['Cabin'].isnull(), 'Cabin_Status'] = 0


# In[58]:


test_dataset.loc[test_dataset['Fare'].isnull(), 'Fare'] = test_dataset['Fare'].mean()


# In[62]:


final_test_data_set = test_dataset.drop(['Name', 'Ticket', 'Cabin'], 1)


# In[66]:


final_test_data_set.head()


# In[64]:


final_data_set.head()


# In[71]:


X_train = final_data_set.iloc[:,2:]
Y_train = final_data_set.iloc[:,1]


# In[73]:


X_test = final_test_data_set.iloc[:,1:]


# In[76]:


from sklearn import linear_model


# In[90]:


# Change categorical values to numberical values using sklearn
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X_train['Sex_Label'] = le.fit_transform(X_train['Sex'])
X_train['Embarked_Label'] = le.fit_transform(X_train['Embarked'])
X_train.head()                


# In[93]:


X_train = X_train.drop(['Sex', 'Embarked'], axis=1)


# In[97]:


X_test['Sex_Label'] = le.fit_transform(X_test['Sex'])
X_test['Embarked_Label'] = le.fit_transform(X_test['Embarked'])
X_test = X_test.drop(['Sex', 'Embarked'], axis=1)
X_test.head()


# In[98]:


X_train.head()


# In[99]:


model = linear_model.LogisticRegression()


# In[100]:


model.fit(X_train, Y_train)


# In[102]:


nparray = model.predict(X_test)


# In[139]:


results = pd.DataFrame(nparray)
results.count()


# In[137]:


results.columns = ['Survived']


# In[140]:


passengerId = final_test_data_set[['PassengerId']]
passengerId.count()


# In[148]:


# pd.DataFrame({"PassengerId": passengerId, "Survived" : results }, ignore_index=True)
submit = pd.concat([passengerId, results], axis=1)
submit.to_csv('Submission.csv', index=False)


# In[ ]:




