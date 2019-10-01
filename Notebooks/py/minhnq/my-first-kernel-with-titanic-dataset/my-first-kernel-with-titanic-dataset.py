#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head(5)


# ## 1. Exploring Data

# In[ ]:


train.isnull().sum()


# **Age** field lost many data  
# **Embarked** just lost 2 cell data so we can easily fill it by the most common data  
# **Cabin** lost the most with around 700 per 900 records

# ### 1.1 Survived feature

# In[ ]:


sb.countplot("Survived", data=train)
plt.show()


# In[ ]:


train['Survived'].mean()


# In[ ]:


train.groupby(['Sex','Pclass']).mean()


# **Female** tended to alive more than **Male**

# In[ ]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True)


# ### 1.2 Pclass feature

# In[ ]:


bar_chart('Pclass')


# Pclass = 3 (the lower class) is more likely to be unsurvived  
# Pclass = 1 (the higher class) has more chance to overcome the disaster

# ### 1.3 Sex feature

# In[ ]:


bar_chart('Sex')


# **Female** has higher chance to survive than **male**

# ### 1.4 Embarked feature

# In[ ]:


bar_chart('Embarked')


# Most of people are come from S, followed by C and Q. The rate of survived/unsurvived people still keep the ratio with number of people from S, C, Q

# ### 1.5 SibSp feature

# In[ ]:


bar_chart('SibSp')


# Going without or only one Sibling or Spouse has higher chance to survive  
# Going without any Sibling or Spouse also more likely to be dead

# ### 1.6 Cabin feature

# In[ ]:


survived = train[train['Survived']==1][train['Cabin'].isnull()==False]['Survived'].value_counts()
dead = train[train['Survived']==0][train['Cabin'].isnull()==False]['Survived'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True)


# People whose cabin numbers are recorded has higher chance to survive

# In[ ]:


survived = train[train['Survived']==1][train['Cabin'].isnull()==True]['Survived'].value_counts()
dead = train[train['Survived']==0][train['Cabin'].isnull()==True]['Survived'].value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True)


# People whose cabin numbers not recorded are more likely to be dead

# ## 2. Feature engineering

# ### 2.1 Name Feature

# In[ ]:


import re
combine=[train,test]
# train_test_df = train.append(test, ignore_index=True)
pattern = re.compile('([A-Za-z]+)\.')
for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, 
                 "Master": 4, "Dr": 5, "Rev": 5, "Col": 5, "Major": 5, "Mlle": 2,"Countess": 4,
                 "Ms": 2, "Lady": 4, "Jonkheer": 5, "Don": 5, "Dona" : 5, "Mme": 3,"Capt": 5,"Sir": 4 }

for dataset in combine:
    #Mr. is 1
    dataset['Title'] = dataset['Title'].replace(['Capt.', 'Col.', 
        'Don.', 'Dr.', 'Major.', 'Rev.', 'Jonkheer.', 'Dona.'], 'Other.')    #Other. is 5  

    dataset['Title'] = dataset['Title'].replace(['Ms.', 'Mlle.'], 'Miss.')   #Miss. is 2

    dataset['Title'] = dataset['Title'].replace('Mme.', 'Mrs.') # Mrs. is 4

    dataset['Title'] = dataset['Title'].replace(['Lady.', 'Master.', 'Countess.', 'Sir.'], 'Royal.') # Mrs. is 4




# In[ ]:


pd.crosstab(train['Title'], train['Sex'])


# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)


# In[ ]:


train.sample(10)


# ### 2.2 Sex Feature

# In[ ]:


sex_mapping = {"male":0 , "female":1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)


# ### 2.3 Ticket Feature

# In[ ]:


# I assume that these 2 features does not affect the result
train.drop('Ticket', axis=1, inplace=True)
test.drop('Ticket', axis=1, inplace=True)


# ### 2.4 Cabin Feature

# In[ ]:


train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))


# In[ ]:


train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)


# In[ ]:


train.sample(10)


# ### 2.5 Embark Feature

# In[ ]:


train = train.fillna({"Embarked": "S"})
test = test.fillna({"Embarked": "S"})


# In[ ]:


embark_mapping = {"S": 1, "C": 2, "Q":3}
train['Embarked'] = train['Embarked'].map(embark_mapping)
test['Embarked'] = test['Embarked'].map(embark_mapping)


# ### 2.6 Age Feature

# In[ ]:


train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


train.loc[ train['Age'] <= 5, 'Age'] = 0, #Baby
train.loc[(train['Age'] > 5) & (train['Age'] <= 12), 'Age'] = 1, #Child
train.loc[(train['Age'] > 12) & (train['Age'] <= 18), 'Age'] = 2, #Teenager
train.loc[(train['Age'] > 18) & (train['Age'] <= 24), 'Age'] = 3, #Student
train.loc[(train['Age'] > 24) & (train['Age'] <= 35), 'Age'] = 4, #Young Adult
train.loc[(train['Age'] > 35) & (train['Age'] <= 60), 'Age'] = 5, #Adult
train.loc[ train['Age'] > 60, 'Age'] = 6 #Senior


# In[ ]:


test.loc[ test['Age'] <= 5, 'Age'] = 0, #Baby
test.loc[(test['Age'] > 5) & (test['Age'] <= 12), 'Age'] = 1, #Child
test.loc[(test['Age'] > 12) & (test['Age'] <= 18), 'Age'] = 2, #Teenager
test.loc[(test['Age'] > 18) & (test['Age'] <= 24), 'Age'] = 3, #Student
test.loc[(test['Age'] > 24) & (test['Age'] <= 35), 'Age'] = 4, #Young Adult
test.loc[(test['Age'] > 35) & (test['Age'] <= 60), 'Age'] = 5, #Adult
test.loc[ test['Age'] > 60, 'Age'] = 6 #Senior


# In[ ]:


train.isnull().sum()


# ### 2.7 Fare Feature

# In[ ]:


# I assume that Fare attribute won't affect much to survivor rate so I will drop it.
test.drop(['Fare'], axis=1, inplace=True)
train.drop(['Fare'], axis=1, inplace=True)


# In[ ]:


train.sample(10)


# ## 3. Modelling

# In[ ]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


# In[ ]:


train_data = train.drop(['Survived', 'PassengerId'], axis=1)
target = train['Survived']


# In[ ]:


train_data.sample(15)


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### 3.1 Using SVM

# In[ ]:


clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ### 3.2 Using Random Forest

# In[ ]:


clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ### 3.3 Using K-NN

# In[ ]:


clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ### 3.4 Using Logistic Regression

# In[ ]:


clf = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ## 4. Testing

# In[ ]:


clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)


# ## 5. References

# - [Titanic Survival Prediction Beginner](https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner)
# - [Titanic Kaggle Solution by Minsuk](https://github.com/minsuk-heo/kaggle-titanic)

# In[ ]:




