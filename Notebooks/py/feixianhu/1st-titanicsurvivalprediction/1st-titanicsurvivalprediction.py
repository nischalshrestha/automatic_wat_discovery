#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# # Load the data set first

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_combine = [df_train, df_test]
df_train.head()


# # Explore the data set by features

# In[ ]:


df_train.info()
print(40*'=')
df_test.info()


# ## 1- Pclass
# there is no missing value

# In[ ]:


df_train[['Survived','Pclass']].groupby('Pclass').mean().sort_values(by='Survived',ascending=False)


# ## 2- Sex
# there is no missing value

# In[ ]:


for df in df_combine:
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df_train[['Sex','Survived']].groupby('Sex').mean()


# ## 3- Name
# there is no missing value, But 'Name' feature needs to be analyzed and categorized. 

# In[ ]:


for df in df_combine:
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(df_train['Title'], df_train['Sex'])


# In[ ]:


pd.crosstab(df_train['Title'], [df_train['Sex'],df_train['Survived']])


# In[ ]:


pd.crosstab(df_test['Title'], df_test['Sex'])


# In[ ]:


for df in df_combine:
    df['Title'] = df['Title'].replace(['Capt','Don','Dona','Jonkheer','Rev'], 'Rare')
    df['Title'] = df['Title'].replace(['Lady','Mlle','Mme','Ms'],             'Miss')
    df['Title'] = df['Title'].replace(['Countess'],                           'Mrs')
    df['Title'] = df['Title'].replace(['Sir'],                                'Mr')
    df['Title'] = df['Title'].replace(['Col','Dr','Major'],                   'Profession')


# In[ ]:


#pd.crosstab(df_train['Title'], df_train['Sex'])
pd.crosstab(df_train['Title'], [df_train['Sex'],df_train['Survived']])


# In[ ]:


df_train[['Title','Survived']].groupby('Title').mean().sort_values(by='Survived',ascending=False)


# now convert the 'Title' to numerical value

# In[ ]:


for df in df_combine:
    df['Title'] = df['Title'].map({'Mrs': 0, 'Miss': 1, 'Master': 2, 'Profession': 3, 'Mr': 4, 'Rare': 5})


# ## 4- Age
# there are some missing inputs
# 
# Considering all the features, 'Age' should be correlated with the following:
# 
# (1)- if have 'Parch'> 0, he/she is married
# 
# (2)- 'Title' will also indicate his/her age approximately
# 
# (3)- 'Pclass'= 1 may be the older people, becasue the ticket is more expensive (can double check with the 'Fare')

# In[ ]:


age_guess = np.zeros((2,3,5)) # Sex-2, Pclass-3, Title-5


# In[ ]:


for df in df_combine:
    df_noAge = df[df['Age'].isnull()]
    for sex in np.arange(0,2): # sex = 0, 1
        for pclass in np.arange(1,4): # pclass = 1, 2, 3
            for title in np.arange(0,5): # title = 0,1,2,3,4
                #only excute the following when 'Age' freature is missing
                if len(df_noAge[(df_noAge.Sex==sex) & (df_noAge.Pclass==pclass) & (df_noAge.Title==title)]):
                    age_guess[sex,pclass-1,title] = int(math.floor(df[(df.Sex==sex) & (df.Pclass==pclass) & (df.Title==title) ]['Age'].dropna().mean())) + 0.5
                    df.loc[df.Age.isnull() & (df.Sex==sex) & (df.Pclass==pclass) & (df.Title==title),'Age'] = age_guess[sex,pclass-1,title]
                    print('guessed age is:',age_guess[sex,pclass-1,title])


# In[ ]:


df_train[['Age','Survived']].groupby('Age').mean()


# In[ ]:


for df in df_combine:
    df['isInfant'] = 0
    df['isKid'] = 0
    df['isOld'] = 0
    df.loc[df['Age']<1,'isInfant']= 1
    df.loc[(df['Age']>=1) & (df['Age']<=6),'isKid'] = 1
    df.loc[df['Age']>=64,'isOld'] = 1


# In[ ]:


df_train[['isInfant','Survived']].groupby('isInfant').mean()


# In[ ]:


df_train[['isKid','Survived']].groupby('isKid').mean()


# In[ ]:


df_train[['isOld','Survived']].groupby('isOld').mean()


# now create the new 'AgeBand' feature

# In[ ]:


for df in df_combine:
    df['tmpt_AgeBand'] = pd.qcut(df['Age'], 4)
df_train.head()
df_train[['tmpt_AgeBand','Survived']].groupby(['tmpt_AgeBand'], as_index=False).mean().sort_values(by='tmpt_AgeBand',ascending=True)


# In[ ]:


df_train.drop(labels='tmpt_AgeBand',inplace=True,axis=1)


# In[ ]:


for df in df_combine:
    df['AgeBand'] = 0
    df.loc[  df.Age <= 21,                 'AgeBand'] = 0
    df.loc[ (df.Age <= 28.5) & (df.Age > 21),'AgeBand'] = 1
    df.loc[ (df.Age <= 36.75) & (df.Age > 28.5),'AgeBand'] = 2
    df.loc[ (df.Age >  36.75),                'AgeBand'] = 3


# ## 5- SibSp & Parch
# 
# SibSp: Siblings and Spouse
# 
# Parch: Parents and Children
# 
# (no missing value)

# In[ ]:


df_train[['SibSp','Survived']].groupby('SibSp').mean().sort_values(by='Survived',ascending=False)


# In[ ]:


df_train[['Parch','Survived']].groupby('Parch').mean().sort_values(by='Survived',ascending=False)


# In[ ]:


for df in df_combine:
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

df_train[['FamilySize','Survived']].groupby('FamilySize').mean().sort_values(by='Survived',ascending=False)


# In[ ]:


for df in df_combine:
    df['isAlone'] = 0
    df.loc[ df['FamilySize']==1 , 'isAlone'] = 1

df_train[['isAlone','Survived']].groupby('isAlone').mean().sort_values(by='Survived',ascending=False)


# In[ ]:


for df in df_combine:
    df['isLargeFamily'] = 0
    df.loc[ df['FamilySize'] > 4, 'isLargeFamily'] = 1

df_train[['isLargeFamily','Survived']].groupby('isLargeFamily').mean().sort_values(by='Survived',ascending=False)


# ## 6-Ticket & Fare
# no missing value for 'Ticket'
# 
# one 'Fare' missing in df_test

# In[ ]:


df_train.head(3)


# In[ ]:


for df in df_combine:
    df['isSpecialTicket'] = df.Ticket.str.extract('([A-Z])', expand=False)
    df['isSpecialTicket'] = df['isSpecialTicket'].fillna('RE')


# In[ ]:


pd.crosstab(df_train['isSpecialTicket'],df_train['Survived'])


# In[ ]:


df_train[['Survived','isSpecialTicket']].groupby('isSpecialTicket').mean().sort_values(by='Survived',ascending=False)


# In[ ]:


for df in df_combine:
    df['isSpecialTicket'] = df['isSpecialTicket'].map({'P':0, 'F':1, 'RE':2, 'C':3, 'S':4, 'L':5, 'W':6, 'A':7})


# In[ ]:


df_test[ df_test.Fare.isnull()]


# In[ ]:


# the missing 'Fare' will be guess by the median value of its Pclass
df_test.loc[df_test.Fare.isnull(),'Fare'] = df_test[df_test['Pclass']==3]['Fare'].median()


# In[ ]:


grid = sns.FacetGrid(df_train, col= 'Survived')
grid.map(plt.hist, 'Fare')
plt.show()


# In[ ]:


df_train['tmpFareBand'] = pd.qcut(df_train['Fare'],4)


# In[ ]:


df_train[['tmpFareBand','Survived']].groupby('tmpFareBand').mean().sort_values(by='Survived',ascending=False)


# In[ ]:


df_train.drop(labels='tmpFareBand',axis=1,inplace=True)
for df in df_combine:
    df['FareBand'] = 0
    df.loc[ (df.Fare <= 7.91),                 'FareBand'] = 0
    df.loc[ (df.Fare<=14.454) & (df.Fare>7.91), 'FareBand'] = 1
    df.loc[ (df.Fare<=31.0) & (df.Fare>14.454),'FareBand'] = 2
    df.loc[ (df.Fare>31.0),                    'FareBand'] = 3


# ## 7- Cabin
# 
# 'Cabin' feature has too many missing value, is not useful, i.e. (891-204)/891*100% (~77%)missing

# In[ ]:


for df in df_combine:
    df['Cabin'] = df['Cabin'].str.extract('([A-Z])', expand=False)
    df['Cabin'] = df['Cabin'].fillna('Z')
df_train[['Cabin','Survived']].groupby('Cabin',as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


for df in df_combine:
    df['Cabin'] = df['Cabin'].map({'D':0, 'E':1, 'B':2, 'F': 3, 'C': 4, 'G': 5, 'A': 6, 'Z': 7, 'T':8})


# ## 8- Embarked

# In[ ]:


pd.crosstab(df_train.Survived, df_train.Embarked)


# In[ ]:


pd.crosstab(df_train.FareBand, df_train.Embarked)


# in each Fare range, the most frequest port is 'S'. So let's just complete all missing 'Embarked' with 'S'

# In[ ]:


for df in df_combine:
    df.loc[ df['Embarked'].isnull(), 'Embarked'] = 'S'


# and then convert to numerical value

# In[ ]:


df_train[['Embarked','Survived']].groupby('Embarked').mean().sort_values(by='Survived',ascending=False)
for df in df_combine:
    df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})


# ## Before the modeling, we take a last look of the dateset

# In[ ]:


df_train.columns


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()
print(40*'=')
df_test.info()


# # Modeling & Predicting with ML

# ### splitting the train and test data

# In[ ]:


selected_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title', 'isInfant','isKid', 'isOld', 'AgeBand', 
                     'FamilySize', 'isAlone', 'isLargeFamily','isSpecialTicket', 'FareBand']
X_train = df_train[selected_features]
Y_train = df_train['Survived']
X_test = df_test[selected_features]
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


X_train.head()


# In[ ]:


X_test.head()


# ### load the ML libraries

# In[ ]:


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


# ### first, using the logistic regression to check the correlations of all features

# In[ ]:


clf = LogisticRegression(C=5)
clf.fit(X_train,Y_train)

df_coeff = pd.DataFrame(X_train.columns.delete(0))
df_coeff.columns = ['Feature']
df_coeff["Correlation"] = pd.Series(clf.coef_[0])

df_coeff.sort_values(by='Correlation', ascending=False)


# ### modeling

# In[ ]:


clfs = [SVC(probability=True), 
        RandomForestClassifier(n_estimators=1000),
        AdaBoostClassifier(n_estimators=1000),
        GradientBoostingClassifier(n_estimators=1000),
        KNeighborsClassifier(n_neighbors=5),
        LogisticRegression(),
       ]


# In[ ]:


score_acc = []
score_f1 = []
for i in np.arange(0,len(clfs)):
    clf = clfs[i]
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_train)
    score_acc.append(accuracy_score(Y_train,Y_pred))
    score_f1.append(f1_score(Y_train,Y_pred))


# In[ ]:


cols = ['SVC','RandomForestClassifier','AdaBoostClassifier','GradientBoostingClassifier','KNeighborsClassifier','LogisticRegression']
scores = pd.DataFrame(columns=cols, index=['accuracy','f1'], data=[score_acc,score_f1])
scores


# ### consider the RandomForestClassifier() for submission due to its highest accuracy & f1 scores

# In[ ]:


clf_chosen = RandomForestClassifier(n_estimators=1000)
clf_chosen.fit(X_train,Y_train)
Y_pred = clf_chosen.predict(X_train)
accuracy_score(Y_train,Y_pred), f1_score(Y_train,Y_pred)


# In[ ]:


Y_test_pred = clf_chosen.predict(X_test)
result_submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': Y_test_pred
})
result_submission.to_csv('submission.csv',index=False)


# In[ ]:




