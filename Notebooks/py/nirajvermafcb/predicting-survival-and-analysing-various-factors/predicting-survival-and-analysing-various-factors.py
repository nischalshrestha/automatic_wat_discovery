#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# ### Getting train and test datsets

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

#train_df.head()
#test_df.head()


# In[ ]:


train_df.info()
print("----------------------------------")
test_df.info()


# **We can drop the PassengerId column, since it is merely an index.It has no correlation with the target variable "Survived"**

# In[ ]:


train_df.drop(['PassengerId'], axis = 1, inplace = True) 
test_df.drop(['PassengerId'], axis = 1, inplace = True) 


# In[ ]:


train_df.head()
#test_df.head()


# **Thus we have drop PassengerID column permanently**

# **We can see that Age,Cabin and Embarked has missing values which we have to deal with.**
# 
# **Age column has significant amount of missing values in both training and testing dataset. We can drop those rows which has missing values but this might result in the loss of some information.Therefore instead of dropping those rows we will rather replace them with a valid value which will generalise better**

# #Age

# In[ ]:



# Checking if any rows has all the null values.If yes then dropping the entire row.

#train_df.dropna(axis=0, how='all')
#test_df.dropna(axis=0, how='all')
#train_df.info()
#print("----------------------------------------")
#test_df.info()


# In[ ]:


train_df[train_df['Age'].isnull()]
train_df[train_df['Age'].isnull()].count()


# In[ ]:


test_df[test_df['Age'].isnull()]
test_df[test_df['Age'].isnull()].count()


# **Thus we got 177 missing age values in training set and 86 missing age values in test set**

# In[ ]:


train_df["Age"].mean()
#train_df["Age"].median()
#train_df['Age'].mode()


# In[ ]:


test_df['Age'].mean()
#test_df["Age"].median()
#test_df['Age'].mode()


# In[ ]:


train_df['Survived'].groupby(pd.qcut(train_df['Age'],6)).mean()


# In[ ]:


pd.qcut(train_df['Age'],6).value_counts()


# In[ ]:


train_df['Embarked'].unique()


# In[ ]:


train_df['Embarked'].value_counts()


# In[ ]:


sns.countplot(train_df['Embarked'])


# In[ ]:


train_df['Survived'].groupby(train_df['Embarked']).mean()


# In[ ]:


sns.countplot(train_df['Embarked'], hue=train_df['Pclass'])


# # Cabin
# **This column has the most nulls (almost 700), but we can still extract information from it, like the first letter of each cabin, or the cabin number.**

# In[ ]:


train_df['Cabin_Letter'] = train_df['Cabin'].apply(lambda x: str(x)[0])


# In[ ]:


train_df['Cabin_Letter'].unique()


# In[ ]:


train_df['Cabin_Letter'].value_counts()


# In[ ]:


train_df['Survived'].groupby(train_df['Cabin_Letter']).mean()


# # Survived

# In[ ]:


train_df['Survived'].value_counts(normalize=True)


# **We can see that nearly 62% of the people in the training set died and 38% survived.**

# In[ ]:


sns.countplot(train_df['Survived'],palette='Set2')


# # Pclass(Passenger Class)
# This variable is very important in determining the survival chances of the passengers.Survival decreases significantly for lower class members as higher class members were given more importance while saving the passengers.

# In[ ]:


train_df['Pclass'].unique()


# **Thus there are 3 classes**

# In[ ]:


train_df['Survived'].groupby(train_df['Pclass']).count()


# In[ ]:


#train_df['Survived'].groupby(train_df['Pclass']).mean()


# In[ ]:


sns.countplot(train_df['Pclass'], hue=train_df['Survived'], palette= 'colorblind')


# #Sex

# **Women and children were given 1st priority.Survival rate of them should be higher**

# In[ ]:


train_df['Sex'].value_counts(normalize=True)


# In[ ]:


#train_df['Survived'].groupby(train_df['Sex']).mean()


# In[ ]:


(train_df['Sex']).value_counts()


# In[ ]:


sns.countplot(train_df['Sex'],palette='cubehelix')


# In[ ]:


train_df['Survived'].groupby(train_df['Sex']).mean()


# # Name

# **Name column does not have any relationship with Survived column.But one thing we can find  useful is the passenger's title**

# In[ ]:


train_df['Name'].head()


# In[ ]:


train_df['Name_Head'] = train_df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train_df['Name_Head'].value_counts()


# In[ ]:


train_df['Survived'].groupby(train_df['Name_Head']).mean()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(train_df['Name_Head'],palette='husl')


# In[ ]:


train_df['Survived'].groupby(train_df['Name_Head']).mean()


# # Sibsp

# In[ ]:


train_df['SibSp'].unique()


# In[ ]:


train_df['SibSp'].value_counts()


# In[ ]:


sns.countplot(train_df['SibSp'],palette='Set1')


# In[ ]:


train_df['Survived'].groupby(train_df['SibSp']).mean()


# # Parch

# In[ ]:


train_df['Parch'].unique()


# In[ ]:


train_df['Parch'].value_counts()


# In[ ]:


sns.countplot(train_df['SibSp'],palette='pastel')


# In[ ]:


train_df['Survived'].groupby(train_df['Parch']).mean()


# # Fare

# **We will fill in the one missing value of Fare in our test set with the mean value of Fare from the training set**

# In[ ]:


test_df['Fare'].fillna(train_df['Fare'].mean(), inplace = True)


# In[ ]:


train_df['Fare'].unique()
train_df['Fare'].min()
train_df['Fare'].max()
#train_df['Fare'].mean()
#train_df['Fare'].mode()


# In[ ]:


pd.qcut(train_df['Fare'], 5).value_counts()


# In[ ]:


train_df['Survived'].groupby(pd.qcut(train_df['Fare'], 5)).mean()


# In[ ]:


pd.crosstab(pd.qcut(train_df['Fare'], 5), columns=train_df['Pclass'])


# **We can see that passenger class and fare variable are highly correlated**

# # Let us perform some feature Engineering

# **This function creates two separate columns: a numeric column indicating the length of a passenger’s Name field, and a categorical column that extracts the passenger’s title.**

# In[ ]:


def names(train, test):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        del i['Name']
    return train, test


# **we impute the null values of the Age column by filling in the mean value of the passenger’s corresponding title and class.**

# In[ ]:


def age_impute(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        data = train.groupby(['Name_Title', 'Pclass'])['Age']
        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))
    return train, test


# **We combine the SibSp and Parch columns into a new variable that indicates family size, and group the family size variable into three categories.**

# In[ ]:


def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        del i['SibSp']
        del i['Parch']
    return train, test


# **This function extract the first letter of the Cabin column.**

# In[ ]:


def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test


# In[ ]:


def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test


# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].mean(), inplace = True)


# **we must convert our categorical columns into dummy variables. The following function does this, and then it drops the original categorical columns.**

# In[ ]:


def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked','Cabin_Letter' 'Name_Title', 'Fam_Size']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test


# **Dropping PassengerId column**

# In[ ]:


def drop(train, test, bye = ['PassengerId']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train, test = names(train, test)
train, test = age_impute(train, test)
train, test = cabin(train, test)
train, test = embarked_impute(train, test)
train, test = fam_size(train, test)
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked','Cabin_Letter', 'Name_Title', 'Fam_Size'])


# In[ ]:


train.drop(['PassengerId'], axis = 1, inplace = True)


# In[ ]:


train.drop( ['Ticket'],axis=1,inplace = True)


# In[ ]:


train.info()


# In[ ]:


len(train.columns)


# In[ ]:


train.head()


# In[ ]:


test.info()


# In[ ]:


test.drop(['PassengerId'], axis = 1, inplace = True)


# In[ ]:


test.drop(['Ticket'], axis = 1, inplace = True)


# In[ ]:


len(test.columns)


# In[ ]:


test.info()


# In[ ]:


test.head()


# # Random forest

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# **Separating predictor variable column and other variable column in training set**

# In[ ]:


rf = RandomForestClassifier(max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)


# In[ ]:


param_grid = { "criterion"   : ["gini", "entropy"],
             "min_samples_leaf" : [1, 5, 10],
             "min_samples_split" : [2, 4, 10, 12, 16],
             "n_estimators": [50, 100, 400, 700, 1000]}


# In[ ]:


gs = GridSearchCV(estimator=rf,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=3,
                  n_jobs=-1
                 )


# In[ ]:


_gs = gs.fit(train.iloc[:, 1:], train.iloc[:, 0])


# In[ ]:


print(gs.best_score_)
print(gs.best_params_)


# In[ ]:


rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=700,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
print ("%.4f" % rf.oob_score_ )


# In[ ]:


predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])


# In[ ]:


test = pd.read_csv(os.path.join('../input', 'test.csv'))
predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
predictions.to_csv('y_test15.csv', sep=",", index = False)


# In[ ]:


# Still workinghhh

