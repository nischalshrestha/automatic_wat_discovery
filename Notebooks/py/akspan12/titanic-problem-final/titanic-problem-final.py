#!/usr/bin/env python
# coding: utf-8

# In[1]:


#loading tests and train data
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[2]:


#data table
train.head()


# In[3]:


#test data
test.head()


# In[4]:


#shape of test and train data
train.shape


# In[5]:



test.shape


# In[6]:


#info about columns in test data
train.info()


# In[7]:


#test set info
test.info()


# In[8]:


#null data summary for train
train.isnull().sum()


# In[9]:


#null data summary for test
test.isnull().sum()


# In[10]:


#library for plotting data
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
# setting seaborn default for plots
sns.set() 


# In[11]:


#function to plot bar chart for survive with relation to other parameters
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))


# In[12]:


bar_chart('Sex')


# 
# women survived more then man

# In[13]:


bar_chart('Pclass')


# people of class 1 survived more whereas that of class 3 are more dead 

# In[14]:


bar_chart('SibSp')


# an alone person is more likely dead 
# but person with more that 2 companion( spouse or sibling) more likely to survive

# In[15]:





bar_chart('Parch')


# alone more likely dead

# In[16]:


bar_chart('Embarked')


# In[17]:


complete_data = [train,test] #combining both data


# In[18]:


for dataset in complete_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.',expand = False)


# In[19]:


train['Title'].value_counts()


# In[20]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in complete_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[21]:


bar_chart('Title')


# 0 - Mr has high chances of death as compare to 1,2- mrs and miss

# In[22]:


train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)


# In[23]:


train.head()


# In[24]:


test.head()


# In[25]:


sex_mapping = {"male":0,"female":1}
for dataset in complete_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# In[26]:


bar_chart('Sex')


# In[27]:


#filling missing age with median age of the title
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[28]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
 
plt.show() 


# 
# converting age to categories
# child: 0
# young: 1
# adult: 2
# mid-age: 3
# senior: 4

# In[29]:


for dataset in complete_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4


# In[30]:


train.head()


# In[31]:


bar_chart('Age')


# In[32]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# more than 50% of 1st class are from S embark
# more than 50% of 2nd class are from S embark
# more than 50% of 3rd class are from S embark
# 
# fill out missing embark with S embark

# In[33]:


for dataset in complete_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[34]:


#mapping embarked
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in complete_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# In[35]:


train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[36]:


test.info()


# In[37]:


train.info()


# In[38]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()  


# In[39]:


for dataset in complete_data:
    dataset.loc[dataset['Fare'] <= 17,'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30),"Fare"] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 93),"Fare"] = 2
    dataset.loc[dataset['Fare'] > 93,"Fare"] = 3


# In[40]:


train.head(20)


# In[41]:


train.Cabin.value_counts()


# In[42]:


#taking only the first character and eliminating the numbers
for dataset in complete_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[43]:


train.head()


# In[44]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[45]:


#mapping cabbin
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in complete_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[46]:


# fill missing cabbin with median for each Pclass
train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# In[47]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1


# In[48]:


facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)


# In[49]:


family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in complete_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[50]:


train.head()


# **dropping unnecessary features **

# In[51]:


features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[52]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape


# In[53]:


test.head(20)


# In[54]:


train.head()


# In[55]:


X_train = train_data


# In[56]:


Y_train = target


# In[57]:


X_train.shape


# In[58]:


Y_train.shape


# In[59]:


X_test  = test.drop("PassengerId", axis=1).copy()


# In[60]:


X_test.head()


# In[61]:


X_test.shape


# **importing models**

# In[62]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[63]:


import numpy as np


# In[64]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# **kNN**

# In[93]:


clf = KNeighborsClassifier(n_neighbors = 10)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[94]:


# kNN Score
round(np.mean(score)*100, 2)


# **Decision Tree**

# In[95]:


clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[96]:


# decision tree Score
round(np.mean(score)*100, 2)


# **Random Forest**

# In[131]:


clf = RandomForestClassifier(n_estimators=11)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[132]:


# Random Forest Score
round(np.mean(score)*100, 2)


# **Naive Bayes**

# In[133]:


clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[134]:


# Naive Bayes Score
round(np.mean(score)*100, 2)


# **SVM**

# In[136]:


clf = SVC(gamma='auto')
scoring = 'accuracy'
score = cross_val_score(clf, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[137]:


#SVM score
round(np.mean(score)*100,2)


# ***Since SVM has the best score we will use it for our prediction***

# In[77]:


clf = SVC(gamma = 'scale')
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)


# **Submission**

# In[78]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

