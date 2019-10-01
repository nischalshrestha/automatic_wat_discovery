#!/usr/bin/env python
# coding: utf-8

# ### Hi, I'm glad to see you all.  
# I'm business and data analyst in Coupang which is one of the largest e-commerce company in Korea.  
# I got a master degree of management engineering  at Ulsan National Institute of Science and Technology.  
# I'm familiar with Econometrics and other statistical approach in social science, but it is the very first time to use machine learning to solve a problem. I hope this will be a good chance to learn machine learning and a part of my portfolio.  
# If you have any question, please leave a comment or give me an e-mail. I'll happy to see you.  
#   
# *Sunmi Yoon, ysunmi0427@gmail.com*

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# ### Import data
# * The passenger 1 whoes name is Braund, Mr. Owen Harris is male, 22 years old. He has 1 SibSp and not Parch. He is not Cabin because the ticket is very cheap and low class, 7.2500. He embarked at Southampton.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(2)


# ### About data type
# * Data type is not very important in this case because there is no column which seems integer but we have to analyze them as string.
# * For example, if the embarkedPort is given as integer, we have to convert that into string because a decision tree I will use interpret integer with their scale, not category.

# In[ ]:


train.dtypes


# ### Summary Statistics
# * There are 891 passengers in training set. 38% are survived fortunately, but 62% of them are died. They are averagely 29.69 years old, too young to die. They have 0.52 siblings / spouses and 0.38 parents / children, 0.90 family in total. Average ticket class is 2.3 and average ticket fare is 32.204. The variation of ticket fare is quite large. You could see that at standard deviation of Fare.

# In[ ]:


train.describe()


# ### Divide total into two, Survived or not.
# * To see feature importance and distribution very quickely, I divide total sample into two sub samples, passenger survived or not. I use matplotlib to visualize them. Through this, I will roughly know which feature is important and which factor affect actual result. 
# * If correlations between features are complex, this simple distribution with divided sample will not help you intuitively.
# * If you have a concept of bayesian probability, you could calculate conditional probability P(woman|survived) and roughy expect posteriori probability in this stage.
# 

# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for col, a in zip(['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'], axes.flatten()):
    a.set_title(col)
    train[train['Survived'] == 1][col].hist(ax=a)


# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for col, a in zip(['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'], axes.flatten()):
    a.set_title(col)
    train[train['Survived'] == 0][col].hist(ax=a)


# ### Preprocessing
# * First of all, I check that there is null values in the dataframe.
# * 2nd, I make dummies with pandas *get_dummies*. It is the same with one hot encoding in scikit learn.
# * 3rd, I could add new features.

# In[ ]:


def nullCol(df):
    return [col for col in list(df.columns) if (df[col].isnull().sum() != 0)]


# * These are columns which have null values. How can we manage this type of passengers?
# * Just remove it? This approach also removes valuable information in other columns, for example, fare and gender.
# * I'll import average value of Age by Sex and port most people in titanic came from instead of null values in train set.
# * There are null in 'Fare' in test set, I'll import average Fare of ticket class s/he had using train set (not test set).

# In[ ]:


print('col which have null values in train set: {}'.format(nullCol(train)))
print('col which have null values in test set: {}'.format(nullCol(test)))


# In[ ]:


# Average age by Sex with pandas groupby
aveAge = train[['Sex', 'Age']].groupby('Sex').mean()
aveFemaleAge = aveAge.at['female', 'Age']
aveMaleAge = aveAge.at['male', 'Age']
aveAge


# In[ ]:


aveFare = train[['Pclass', 'Fare']].groupby('Pclass').mean()
aveFirstFare = aveFare.at[1, 'Fare']
aveSecondFare = aveFare.at[2, 'Fare']
aveThirdFare = aveFare.at[3, 'Fare']
aveFare


# In[ ]:


# Frequency with collections.Counter()
import collections
ports = collections.Counter(train['Embarked'])
mostPopularPort = max(ports, key=ports.get)
ports


# In[ ]:


# Insert average age by Sex instead of null
train.loc[(train['Age'].isnull()) & (train['Sex'] == 'female'), 'Age'] = aveFemaleAge
train.loc[(train['Age'].isnull()) & (train['Sex'] == 'male'), 'Age'] = aveMaleAge
test.loc[(test['Age'].isnull()) & (test['Sex'] == 'female'), 'Age'] = aveFemaleAge
test.loc[(test['Age'].isnull()) & (test['Sex'] == 'male'), 'Age'] = aveMaleAge

# Insert port which is most people on Titanic came from instead of null
train.loc[(train['Embarked'].isnull()), 'Embarked'] = mostPopularPort

# Insert average fare by ticket class instead of null
test.loc[(test['Fare'].isnull()) & (test['Pclass'] == 1), 'Fare'] = aveFirstFare
test.loc[(test['Fare'].isnull()) & (test['Pclass'] == 2), 'Fare'] = aveSecondFare
test.loc[(test['Fare'].isnull()) & (test['Pclass'] == 3), 'Fare'] = aveThirdFare


# In[ ]:


# make dummy and erase previous column
def addDummies(df, cols):
    for col in cols:
        dum = pd.get_dummies(df[col])
        dum.columns = [i + col for i in dum.columns]
        df = df.join(dum).drop([col], axis=1)
    return df


# In[ ]:


train = addDummies(train, ['Sex', 'Embarked'])
test = addDummies(test, ['Sex', 'Embarked'])


# ### New Features
# * If certain cabin is close to the life boat, we could expect people in that cabin have high probability to live than others.
# * I use cabin as category information, for example, A95 -> A.
# * If a passenger have a peer / peers, they might help each other to get a boat or warn a disaster earlier than others. 
# * I use ticket number to find people who board on Titanic with someone else, not alone.
# * I consider whole family size as one of features.
# 
# * If s/he has young child, s/he might allows to board on a life boat than others who have no child because s/he has responsibility to protect them.
# * I use both parch, age, and names to find people who have young child if I have enough time.

# In[ ]:


# all cabin in training set
train['Cabin'].unique()

# all cabin in test set
# test['Cabin'].unique()


# In[ ]:


import numpy as np
train['Cabin'] = train['Cabin'].apply(lambda x: 'Z' if x is np.nan else x[0])
test['Cabin'] = test['Cabin'].apply(lambda x: 'Z' if x is np.nan else x[0])


# In[ ]:


train = addDummies(train, ['Cabin'])
test = addDummies(test, ['Cabin'])


# In[ ]:


# consider whole family size
train['Family'] = train['SibSp'] + train['Parch']
train['Family'] = train['SibSp'] + train['Parch']


# In[ ]:


# I use ticket number to find people who board on Titanic with someone else, not alone.
withPeer = train.groupby('Ticket').count()['PassengerId'].reset_index()
withPeer['peer'] = withPeer['PassengerId'] > 1
train = train.merge(withPeer[['Ticket', 'peer']], on='Ticket')

withPeer = test.groupby('Ticket').count()['PassengerId'].reset_index()
withPeer['peer'] = withPeer['PassengerId'] > 1
test = test.merge(withPeer[['Ticket', 'peer']], on='Ticket')


# In[ ]:


train.columns


# ### Decision Tree Ensembles / Random Forest

# In[ ]:


indep = [col for col in test.columns if col not in ['PassengerId', 'Survived', 'Name', 'Ticket']]


# In[ ]:


indep


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

ranfor = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(ranfor, train[indep], train['Survived'])
print("cross validation score: {}".format(scores))


# In[ ]:


trees = ranfor.fit(train[indep], train['Survived'])
pred = trees.predict(test[indep])


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pred
    })

submission.to_csv('./submission.csv', index=False)

