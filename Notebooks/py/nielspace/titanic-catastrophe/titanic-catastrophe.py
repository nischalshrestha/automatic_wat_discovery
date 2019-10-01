#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[2]:


import pandas as pd


# In[3]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
test_for_prediction=pd.read_csv('../input/test.csv')


# # Get Info

# In[4]:


train.head()


# In[5]:


train.isnull().sum()


# In[6]:


train.info()


# In[7]:


train.describe()


# # Visualize

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import style


# In[9]:


def bar_chart (feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead=train[train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index=['Suvived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
    
    


# In[10]:


bar_chart('SibSp')


# In[11]:


bar_chart('Pclass')


# In[ ]:





# # Feature Engineering

# In[12]:


data=[train,test]
for dataset in data:
    dataset['Title']=dataset['Name'].str.extract('([a-zA-Z]+)\.',expand=False)


# In[13]:


train.head()


# In[14]:


train['Title'].value_counts()


# In[15]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[16]:


train=train.drop(['Name'],axis=1)
test=test.drop(['Name'],axis=1)
train=train.drop(['PassengerId'],axis=1)
test=test.drop(['PassengerId'],axis=1)
test=test.drop(['Cabin'],axis=1)
train=train.drop(['Cabin'],axis=1)
test=test.drop(['Ticket'],axis=1)
train=train.drop(['Ticket'],axis=1)


# In[17]:


data=[train,test]
Sex_mapping={'male':0,'female':1}
for dataset in data:
    dataset['Sex']=dataset['Sex'].map(Sex_mapping)


# In[18]:


train=train.fillna(train.mean())
test=test.fillna(test.mean())


# In[19]:


test.head()


# In[20]:


data=[train,test]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[21]:


data=[train,test]
for dataset in data:
    dataset['Fare']=dataset['Fare'].astype(int)
    dataset.loc[dataset['Fare']<10, 'Fare']=0
    dataset.loc[(dataset['Fare']>10) & (dataset['Fare']<=20), 'Fare']=1
    dataset.loc[(dataset['Fare']>20) & (dataset['Fare']<=30), 'Fare']=2
    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=40), 'Fare']=3
    dataset.loc[(dataset['Fare']>40) & (dataset['Fare']<=50), 'Fare']=4
    dataset.loc[(dataset['Fare']>50) & (dataset['Fare']<=60), 'Fare']=5
    dataset.loc[dataset['Fare']>60, 'Fare']=6


# In[ ]:





# In[22]:


data=[train,test]
common_value='S'
for dataset in data:
    dataset['Embarked']=dataset['Embarked'].fillna(common_value)


# In[23]:


ports={'S':0,'Q':1,'C':2}
data=[train,test]
for dataset in data:
    dataset['Embarked']=dataset['Embarked'].map(ports)


# In[24]:


bar_chart('Title')


# # Modeling

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np


# In[26]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']


# In[27]:


train_data.shape, target.shape


# # KFold

# In[28]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# # KNN

# In[29]:


clf = KNeighborsClassifier(n_neighbors = 13)
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1)
print(score)


# In[30]:


round(np.mean(score)*100,4)


# # Decision_Tree

# In[31]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


clf_2 = DecisionTreeClassifier()
scoring = 'accuracy'
score_2 = cross_val_score(clf_2, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score_2)


# In[36]:


round(np.mean(score_2)*100,2)


# # Random_Forrest

# In[37]:


clf_3 = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score_3 = cross_val_score(clf_3, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score_3)


# In[38]:


round(np.mean(score_3)*100,1)


# # SVM

# In[39]:


clf_4 = SVC()
scoring = 'accuracy'
score_4 = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score_4)


# In[40]:


round(np.mean(score_4)*100,4)


# In[ ]:





# In[41]:


clf_5 = LogisticRegression()
scoring = 'accuracy'
score_5 = cross_val_score(clf_5, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score_5)


# In[42]:


round(np.mean(score_5)*100,3)


# In[ ]:





# In[43]:


from sklearn.naive_bayes import GaussianNB


# In[44]:


clf_6=GaussianNB()
scoring="accuracy"
score_6=cross_val_score(clf_6, train_data,target,cv=k_fold, n_jobs=1,scoring=scoring)


# In[45]:


round(np.mean(score_6)*100,2)


# # Testing

# In[46]:


from sklearn.svm import SVC


# In[47]:


clf = LogisticRegression()
clf.fit(train_data, target)


# In[ ]:





# In[48]:


test_data = test
prediction = clf.predict(test_data)


# In[49]:


prediction


# # Submission

# In[50]:


sub_5=pd.DataFrame({"PassengerId": test_for_prediction["PassengerId"],'Survived':prediction })
sub_5.to_csv('sub_5_6.csv', index=False)


# In[51]:


test.head(4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




