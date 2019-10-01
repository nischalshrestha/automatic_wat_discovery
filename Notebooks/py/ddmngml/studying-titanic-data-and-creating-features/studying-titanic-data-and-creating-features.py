#!/usr/bin/env python
# coding: utf-8

# # Excercises with the Titanic Dataset

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
#df = pd.read_csv('train.csv')
#df_test = pd.read_csv('test.csv')

print("Cols: ", df.columns)

print("Cab", len(df[df['Cabin'].isnull()][['Cabin', 'Name' ]]))
print("Emb", len(df[df['Embarked'].isnull()][['Embarked', 'Name' ]]))
print("Sex", len(df[df['Sex'].isnull()][['Sex', 'Name' ]]))
print("Age", len(df[df['Age'].isnull()][['Age', 'Name' ]]))
print("Par", len(df[df['Parch'].isnull()][['Parch', 'Name' ]]))
print("Sib", len(df[df['SibSp'].isnull()][['SibSp', 'Name' ]]))
print("Far", len(df[df['Fare'].isnull()][['Fare', 'Name' ]]))
print("Tic", len(df[df['Ticket'].isnull()][['Ticket', 'Name' ]]))
print("Pcl", len(df[df['Pclass'].isnull()][['Pclass', 'Name' ]]))

# Filling NAs
df["Embarked"] = df["Embarked"].fillna('C')

# Fill missing fields with columns means
df = df.fillna(df.mean())
df['Cabin'] = df['Cabin'].fillna('U')

# Fill missing fields with columns means
df_test = df_test.fillna(df_test.mean())
df_test['Cabin'] = df_test['Cabin'].fillna('U')


# Extracting numeric part from tickets and creating a new feature
ticketnos = []
for s in df['Ticket']:
    ticketnos.append(''.join([n for n in s.split() if n.isdigit()]))
df['TicketNo'] = pd.to_numeric(pd.Series(ticketnos))
df['TicketNo'] = df['TicketNo'].fillna(df['TicketNo'].median())

ticketnos = []
for s in df_test['Ticket']:
    ticketnos.append(''.join([n for n in s.split() if n.isdigit()]))
df_test['TicketNo'] = pd.to_numeric(pd.Series(ticketnos))


print(df.describe())
print(df.dtypes)

# Transforming cabin code to a deck, adding 'U' (unknown) for the missing ones
df['Deck'] = pd.Series([re.split('(\d.*)',s)[0][0] for s in df['Cabin']])
df_test['Deck'] = pd.Series([re.split('(\d.*)',s)[0][0] for s in df_test['Cabin']])


# 

# In[ ]:


#--------------------
# Under-18 feature
df['U18'] = df['Age'] < 18
df_test['U18'] = df_test['Age'] < 18

bins = [0, 18, 23, 55, 80]
df['AgeGroup'] = pd.cut(df['Age'], bins)
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins)

sns.factorplot(x="AgeGroup", y="Survived", data=df)
print(df["AgeGroup"].unique())

#--------------------
# Family size
df['FamilySize'] = (df['Parch'] + df['SibSp'])
df_test['FamilySize'] = (df_test['Parch'] + df_test['SibSp'])

bins = [-1, 2, 5, 7, 11]
df['FamilySizeGroup'] = pd.cut(df['FamilySize'], bins)
df_test['FamilySizeGroup'] = pd.cut(df_test['FamilySize'], bins)

sns.factorplot(x="FamilySizeGroup", y="Survived", data=df)
print(df["FamilySizeGroup"].unique())

#--------------------
# Name length
df['NameLen'] = [len(n) for n in df['Name']]
df_test['NameLen'] = [len(n) for n in df_test['Name']]

bins = [0, 20, 40, 57, 85]
df['NameLenGroup'] = pd.cut(df['NameLen'], bins)
df_test['NameLenGroup'] = pd.cut(df_test['NameLen'], bins)

sns.factorplot(x="NameLenGroup", y="Survived", data=df)
print(df["NameLenGroup"].unique())
#--------------------


# 

# In[ ]:


titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.', 'Col.', 'Capt.', 'Sir.', 'Lady.', 'Countess.', 'Dona.'
          , 'Major.', 'Don.', 'Rev.', 'Father', 'Jonkheer.', 'Mlle.', 'Ms.', 'Mme.']

df['Title'] = df['Name'].apply(lambda n: str(set([w for w in n.split()]) & set(titles)) )
df_test['Title'] = df_test['Name'].apply(lambda n: str(set([w for w in n.split()]) & set(titles)) )

df['Title'].unique()
df_test['Title'].unique()

#df['Name'][df['Title']=='set()']
#df_test['Name'][df_test['Title']=='set()']


# 

# In[ ]:



labels = ['Sex', 'Embarked', 'Deck', 'NameLenGroup', 'FamilySizeGroup', 'AgeGroup', 'Title']
les = {}

for l in labels:
    print('labeling ' + l)
    les[l] = LabelEncoder()
    #print(df[l])
    les[l].fit(df[l].append(df_test[l]))
    tr = les[l].transform(df[l]) 
    df.loc[:, l + '_feat'] = pd.Series(tr, index=df.index)

    tr_test = les[l].transform(df_test[l]) 
    df_test.loc[:, l + '_feat'] = pd.Series(tr_test, index=df_test.index)


#print(df.head())


# 

# In[ ]:


X_train = df.drop(labels, 1)     .drop('Survived', 1)     .drop('Cabin', 1)     .drop('Ticket', 1)     .drop('NameLen', 1)     .drop('Name', 1)     .drop('PassengerId', 1)
y_train = df['Survived']

X_test = df_test.drop(labels, 1)     .drop('Cabin', 1)     .drop('Ticket', 1)     .drop('NameLen', 1)     .drop('Name', 1)     .drop('PassengerId', 1)

print("X_train shape", X_train.shape)
print("X_test  shape", X_test.shape)

#X_train.describe()
#X_test.describe()


# 

# In[ ]:



full_set = X_train[:]
full_set['Survived'] = y_train

plt.title('Pearson Correlation for training set')
sns.heatmap(full_set.astype(float).corr(),
            linewidths=0.1,
            vmax=1.0, 
            square=True, 
            cmap="PuBuGn", 
            linecolor='w', 
            annot=False)

full_set.corr()['Survived'].abs().sort_values(ascending = False)


# 

# In[ ]:


X_train = X_train.drop('SibSp', 1)     .drop('Parch', 1) 

X_test = X_test.drop('SibSp', 1)     .drop('Parch', 1) 


# In[ ]:


def dummies(train, test, columns ):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test
X_train, X_test = dummies(X_train, X_test, columns=['Pclass'
                                                    , 'Sex_feat'
                                                    , 'Embarked_feat'
                                                    , 'Deck_feat'
                                                    , 'TicketNo'
                                                    , 'Title_feat'
                                                    , 'AgeGroup_feat'
                                                    , 'FamilySizeGroup_feat'
                                                    , 'NameLenGroup_feat'])


# In[ ]:


full_set = X_train[:]
full_set['Survived'] = y_train

full_set.corr()['Survived'].abs().sort_values(ascending = False)


# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import math

X_tr, X_ts, y_tr, y_ts = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
print(X_tr.shape, y_tr.shape, X_ts.shape, y_ts.shape)


# 

# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


#forest = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
#param_grid = { "criterion" : ["gini", "entropy"]
#              , "min_samples_leaf" : [1, 5, 10]
#              , "min_samples_split" : [2, 4, 10, 12, 16]
#              , "n_estimators": [25, 50, 100, 400, 700]}
#gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring='accuracy', cv=3, n_jobs=-1)
#gs = gs.fit(X_tr, y_tr)


# In[ ]:


#print(gs.best_score_)
#print(gs.best_params_)
#print(gs.cv_results_)


# In[ ]:


rf = RandomForestClassifier( criterion='entropy', 
                             n_estimators=400,
                             min_samples_split=16,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

rf.fit(X_tr, y_tr)
pred = rf.predict(X_ts)

score = rf.score(X_ts, y_ts)
err = math.sqrt(((pred - y_ts)**2).mean())
print("Error: %.3f Score: %.3f" % (err, score))


# 

# In[ ]:


pd.concat((pd.DataFrame(X_train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]


# 

# In[ ]:


# Training the validated model with the whole training set
rf.fit(X_train, y_train)

pred = rf.predict(X_test)

df_test['Survived'] = pd.Series(pred)
sub = df_test[['PassengerId','Survived']]

sub.to_csv('submission_forest.csv', index=False)

