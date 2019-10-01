#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# ##  **개관**

# 입문자들에게 언제나 추천되는** Titanic **에 관한 데이터 분석 문제를 풀어보겠습니다.   
# 이 문제는 [케글](https://www.kaggle.com/) 에 기본문제로 등록되어 있습니다. 링크는 여기.  https://www.kaggle.com/c/titanic 
# 
# 데이터분석 문제의 가장 큰 범주로 **Classification** 과 **Regression**이 있습니다.  
# 그 중에 이 **Titanic** 문제는 **Classification(분류)**를 하는 문제가 되겠습니다.

# 이 문제를 푸는데 있어서 아래와 같은 순서로 풀고자 하였습니다.  
# 
# 1. 데이터를 관찰하고
# 2. 데이터를 정제하고
# 3. 모형에 적용하여 정답 도출.
# 
# 또 이 notebook 파일을 작성하는데에 아래의 kernel들을 적극적으로 참고하였습니다.
# 
# https://www.kaggle.com/datalana/the-preprocessing-framework-with-titanic  
# https://www.kaggle.com/anshumank/titanic-survivor-prediction  
# https://www.kaggle.com/athabascaai/titanic-passenger-graphs-for-feature-inspection
# 

# 목차는 아래와 같습니다. 링크를 걸어두었고 클릭하여 이동가능합니다.
# 
# **Table Of Contents**
# 
# 1. [페키지 Load, 데이터 Load](#'1')  
#     1.1 [페키지 Load](#'1.1'')  
#     1.2 [데이터 Load](#'1.2')  
# 2. [데이터 관찰](#'2')  
#     2.1 [Data Description](#'2.1')   
#     2.2 [데이터 상세](#'2.2')  
#     2.3 [여러 그래프](#'2.3')  
# 3. [데이터 정제](#'3')  
#     3.1 [Null값 처리](#'3.1')  
#     3.2 [zero variance 처리 ](#'3.2')  
#     3.3 [one-hot-coding](#'3.3')  
#     3.4 [feature engineering](#'3.4')  
# 4. [모형적용](#'4')    
#     4.1 [Random Forest](#'4.1')  
#     4.2 Support Vector Machine  
#     4.3 Neural-Net  
# 5. [성능평가  ](#'5')   
#     5.1 [Random Forest](#'5.1')   
# 

# ## <a id = '1'>1. 페키지 Load, 데이터 Load</a>

# ### <a id ='1.1'>1.1 페키지 Load </a>

# In[205]:



import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', color_codes=True)
palette = 'cubehelix' # good for bw printer

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# ### <a id ='1.2'> 1.2 데이터 Load </a>

# In[206]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.head()


# ## <a id = '2'>2. 데이터 관찰 </a>

# ### <a id = '2.1'>2.1 Data Description</a>

# https://www.kaggle.com/c/titanic/data 을 보면 각 데이터 타입에관한 설명이 있습니다.   
# Data Dictionary와, Variable Notes.
# 
# 데이터분석의 기본은 데이터의 관찰이고,   
# 어느정도 수준이 쌓이고나면 알고리즘의 성능은 이 관찰에서부터 시작됩니다.!  
# 아래를 천천히 읽어보도록 합니다.

# **Data Dictionary**
# 
# Variable |	Definition | Key
#  ---- | ---- | ----
# survival |	Survival	| 0 = No, 1 = Yes
# pclass	| Ticket class	| 1 = 1st, 2 = 2nd, 3 = 3rd
# sex|	Sex	|
# Age|	Age in years	|
# sibsp	|# of siblings / spouses aboard the Titanic	|
# parch	|# of parents / children aboard the Titanic	|
# ticket|	Ticket number	|
# fare	|Passenger fare	|
# cabin|	Cabin number	|
# embarked	|Port of Embarkation|	C = Cherbourg, Q = Queenstown, S = Southampton
# 

# **Variable Notes**
# 
# pclass: A proxy for socio-economic status (SES)  
# 1st = Upper  
# 2nd = Middle  
# 3rd = Lower  
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5  
# sibsp: The dataset defines family relations in this way...  
# Sibling = brother, sister, stepbrother, stepsister  
# Spouse = husband, wife (mistresses and fiancés were ignored)  
# parch: The dataset defines family relations in this way...  
# Parent = mother, father  
# Child = daughter, son, stepdaughter, stepson  
# Some children travelled only with a nanny, therefore parch=0 for them.  

# ### <a id = '2.2'>2.2 데이터 상세</a>

# In[207]:


train_df.info()


# ### <a id = '2.3'>2.3 여러 그래프</a>

# In[208]:


survived_sex = train_df[train_df['Survived']==1]['Sex'].value_counts()
dead_sex = train_df[train_df['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5), color=['pink','blue'], title='Survival by the Sex')


# In[209]:


df = train_df.copy()
df['Age'].fillna(df['Age'].median(), inplace=True)

figure = plt.figure(figsize=(10,5))
plt.hist([df[df['Survived']==1]['Age'],df[df['Survived']==0]['Age']], 
         stacked=True, color = ['g','r'], 
         bins = 30, label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.title('Survival by Age')


# In[210]:


df = train_df.copy()
df['Fare'].fillna(df['Fare'].median(), inplace=True)
figure = plt.figure(figsize=(10,5))
plt.hist(
            [df[df['Survived']==1]['Fare'],
            df[df['Survived']==0]['Fare']], 
            stacked=True, color = ['g','r'],
            bins = 30,
            label = ['Survived','Dead']
        )
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.title('Survival by Ticket Price')


# ## <a id = '3'> 3. 데이터 정제 </a>

# ### <a id = '3.1'> 3.1 Null값 처리. </a>

# In[211]:


train_df.isnull().sum().sort_values(ascending=False)


# In[212]:


test_df.isnull().sum().sort_values(ascending=False)


# In[213]:


del train_df['Cabin']
# del train_df['Name']
# del train_df['PassengerId']
del train_df['Ticket']
del test_df['Cabin']
# del test_df['Name']
# del test_df['PassengerId']
del test_df['Ticket']

train_df=train_df.dropna(subset = ['Embarked'])
test_df=test_df.dropna(subset = ['Embarked'])


# In[214]:


train_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(train_df['Fare'].median(), inplace=True)
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)


# In[215]:


train_df.info()


# In[216]:


train_df.isnull().sum().sort_values(ascending=False)


# ### <a id = '3.2'> 3.2 zero variance 처리. </a>

# In[217]:


train_df['Title'] = train_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
test_df['Title'] = test_df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[218]:


train_df['Title'].value_counts()


# In[219]:


test_df['Title'].value_counts()


# In[220]:


for i in range(len(test_df)):
    if test_df.loc[i,'Title'] not in list(set(list(train_df['Title']))):
        test_df.loc[i,'Title'] ='Don'


# In[221]:


titles = (train_df['Title'].value_counts() < 10)
train_df['Title'] = train_df['Title'].apply(lambda x: 'other' if titles.loc[x] == True else x)
test_df['Title'] = test_df['Title'].apply(lambda x: 'other' if titles.loc[x] == True else x)

train_df['Title'].value_counts()


# In[222]:


test_df['Title'].value_counts()


# In[223]:


del train_df['Name']
del test_df['Name']


# In[224]:


train_df.head()


# In[225]:


train_df.info()


# ### <a id = '3.3'> 3.3 one-hot-coding </a> 

# In[226]:


train_dummies = pd.get_dummies(train_df)
test_dummies = pd.get_dummies(test_df)


# In[227]:


train_dummies.info()


# In[228]:


train_dummies.head()


# ### <a id = '3.4'> 3.4 feature engineering </a>

# 데이터를 충분한 시간 관찰 한 후에. 하도록 하겠습니다.

# ## <a id = '4'>4.모형적용 </a>

# ### <a id = '4.1'>4.1 Random Forest</a>

# In[229]:


input_list = list(train_dummies.columns)
input_list.remove('Survived')
input_list.remove('PassengerId')


# In[230]:


input_list


# In[231]:


X_train = train_dummies[input_list]
Y_train = train_dummies['Survived']
X_test = test_dummies[input_list]
print(len(X_test))
# Y_test = test_dummies['Survived']


# In[232]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[233]:


classifier = RandomForestClassifier(n_estimators = 80, criterion = 'entropy', max_depth = 6)
classifier.fit(X_train,Y_train)


# In[234]:


Y_train_pred = classifier.predict(X_train)
Y_test_pred = classifier.predict(X_test)


# In[ ]:





# ## <a id ='5'> 5. 성능평가 </a>

# ### <a id = '5.1'> Random Forest </a>

# In[235]:


confusion_matrix(Y_train, Y_train_pred)


# In[236]:


len(X_test)


# In[238]:


len(test_dummies['PassengerId'])


# In[239]:


result = pd.DataFrame({'PassengerId' : test_dummies['PassengerId'],
                       'Survived' : Y_test_pred})


# In[241]:


result.to_csv('result.csv',index=False)


# In[ ]:




