#!/usr/bin/env python
# coding: utf-8

# Import needed tools

# In[ ]:


import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Import dataset

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.describe(include="all")


# In[ ]:


test_df.describe(include="all")


# Take a look at the correlation

# In[ ]:


from string import ascii_letters
import seaborn as sns
corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# Check the null value

# In[ ]:


print(pd.isnull(train_df).sum())


# In[ ]:


train_df.head()


# Drop unhelpful attribute

# In[ ]:


train_df = train_df.drop(['PassengerId'], axis=1)


# In[ ]:


print(pd.isnull(train_df).sum())


# In[ ]:


print(pd.isnull(test_df).sum())


# Impute and transform dummy variables

# In[ ]:


train_df["Embarked"] = train_df["Embarked"].fillna("S")


# In[ ]:


#embark_dummies_train  = pd.get_dummies(train_df['Embarked'])


# In[ ]:


#embark_dummies_test  = pd.get_dummies(test_df['Embarked'])


# In[ ]:


#train_df = train_df.join(embark_dummies_train)
#test_df = test_df.join(embark_dummies_test)

#train_df.drop(['Embarked'], axis=1,inplace=True)
#test_df.drop(['Embarked'], axis=1,inplace=True)


# In[ ]:


print(pd.isnull(train_df).sum())


# In[ ]:


print(pd.isnull(test_df).sum())


# In[ ]:


test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)


# In[ ]:


train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)


# A precise way to impute Age !!! Learn from https://www.kaggle.com/ash316/eda-to-prediction-dietanic/notebook

# In[ ]:


train_df['Initial']=0
for i in train_df:
    train_df['Initial']=train_df.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(train_df.Initial,train_df.Sex).T.style.background_gradient(cmap='summer_r')


# In[ ]:


train_df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


train_df.groupby('Initial')['Age'].mean()


# In[ ]:


train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mr'),'Age']=33
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mrs'),'Age']=36
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Master'),'Age']=5
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Miss'),'Age']=22
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Other'),'Age']=46


# In[ ]:


plt.hist(train_df.Age)


# In[ ]:


train_df['Age_band']=0
train_df.loc[train_df['Age']<=16,'Age_band']=0
train_df.loc[(train_df['Age']>16)&(train_df['Age']<=24),'Age_band']=1
train_df.loc[(train_df['Age']>24)&(train_df['Age']<=32),'Age_band']=2
train_df.loc[(train_df['Age']>32)&(train_df['Age']<=48),'Age_band']=3
train_df.loc[(train_df['Age']>48)&(train_df['Age']<=64),'Age_band']=4
train_df.loc[train_df['Age']>64,'Age_band']=5


# In[ ]:


test_df['Initial']=0
for i in test_df:
    test_df['Initial']=test_df.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(test_df.Initial,test_df.Sex).T.style.background_gradient(cmap='summer_r')


# In[ ]:


test_df['Initial'].replace(['Ms','Dr','Col','Rev','Sir','Dona'],['Miss','Mr','Other','Other','Mr','Other'],inplace=True)


# In[ ]:


train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mr'),'Age']=33
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Mrs'),'Age']=36
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Master'),'Age']=5
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Miss'),'Age']=22
train_df.loc[(train_df.Age.isnull())&(train_df.Initial=='Other'),'Age']=46


# In[ ]:


plt.hist(test_df.Age,range=(test_df.Age.min(),test_df.Age.max()))


# In[ ]:


test_df['Age_band']=0
test_df.loc[test_df['Age']<=16,'Age_band']=0
test_df.loc[(test_df['Age']>16)&(test_df['Age']<=24),'Age_band']=1
test_df.loc[(test_df['Age']>24)&(test_df['Age']<=32),'Age_band']=2
test_df.loc[(test_df['Age']>32)&(test_df['Age']<=48),'Age_band']=3
test_df.loc[(test_df['Age']>48)&(test_df['Age']<=64),'Age_band']=4
test_df.loc[test_df['Age']>64,'Age_band']=5


# In[ ]:


corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# In[ ]:


train_df['Family_Size']=0
train_df['Family_Size']=train_df['Parch']+train_df['SibSp']


# In[ ]:


test_df['Family_Size']=0
test_df['Family_Size']=test_df['Parch']+test_df['SibSp']


# In[ ]:


train_df['Alone']=0
test_df['Alone']=0
train_df.loc[train_df.Family_Size==0,'Alone']=1
test_df.loc[test_df.Family_Size==0,'Alone']=1


# In[ ]:


corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# In[ ]:


train_df['Fare_Range']=pd.qcut(train_df['Fare'],4)
train_df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# In[ ]:


train_df['Fare_new']=0
train_df.loc[train_df['Fare']<=7,'Fare_new']=0
train_df.loc[(train_df['Fare']>7)&(train_df['Fare']<=14),'Fare_new']=1
train_df.loc[(train_df['Fare']>14)&(train_df['Fare']<=31),'Fare_new']=2
train_df.loc[(train_df['Fare']>31)&(train_df['Fare']<=512),'Fare_new']=3

test_df['Fare_new']=0
test_df.loc[test_df['Fare']<=7,'Fare_new']=0
test_df.loc[(test_df['Fare']>7)&(test_df['Fare']<=14),'Fare_new']=1
test_df.loc[(test_df['Fare']>14)&(test_df['Fare']<=31),'Fare_new']=2
test_df.loc[(test_df['Fare']>31)&(test_df['Fare']<=512),'Fare_new']=3


# I tried to do the same thing to Cabin but failed, since the value in test dataset is different

# In[ ]:


mode = lambda x: x.mode() if len(x) > 2 else np.array(x)
train_df.groupby('Initial')['Cabin'].agg(mode)


# In[ ]:


train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Mr'),'Cabin']='B51'
train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Mrs'),'Cabin']='D'
train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Master'),'Cabin']='F2'
train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Miss'),'Cabin']='E101'
train_df.loc[(train_df.Cabin.isnull())&(train_df.Initial=='Other'),'Cabin']='A26'


# In[ ]:


test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Mr'),'Cabin']='B51'
test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Mrs'),'Cabin']='D'
test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Master'),'Cabin']='F2'
test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Miss'),'Cabin']='E101'
test_df.loc[(test_df.Cabin.isnull())&(test_df.Initial=='Other'),'Cabin']='A26'


# In[ ]:


train_df = train_df.drop(['Age'], axis=1)
test_df    = test_df.drop(['Age'], axis=1)


# In[ ]:


corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# Transform dummy variables again

# In[ ]:


train_df['Embarked'] = train_df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)
test_df['Embarked'] = test_df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)


# In[ ]:


train_df = pd.get_dummies(train_df, columns=['Embarked','Initial','Parch','SibSp','Pclass'] )
test_df = pd.get_dummies(test_df, columns=['Embarked','Initial','Parch','SibSp','Pclass'] )


# In[ ]:


#pclass_dummies_train = pd.get_dummies(train_df['Pclass'])
#pclass_dummies_train.columns = ['Class_1','Class_2','Class_3']

#pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
#pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']


# In[ ]:


#train_df.drop(['Pclass'],axis=1,inplace=True)
#test_df.drop(['Pclass'],axis=1,inplace=True)
#train_df = train_df.join(pclass_dummies_train)
#test_df = test_df.join(pclass_dummies_test)


# In[ ]:


train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_df['Sex'] = test_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


#pclass_dummies_train = pd.get_dummies(train_df['Sex'])
#pclass_dummies_train.columns = ['Sex1','Sex2']
#pclass_dummies_test = pd.get_dummies(test_df['Sex'])
#pclass_dummies_test.columns = ['Sex1','Sex2']
#train_df.drop(['Sex'],axis=1,inplace=True)
#test_df.drop(['Sex'],axis=1,inplace=True)
#train_df = train_df.join(pclass_dummies_train)
#test_df = test_df.join(pclass_dummies_test)


# In[ ]:


corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# In[ ]:


#train_df = train_df.drop(['Sex1'], axis=1)
#test_df = test_df.drop(['Sex1'], axis=1)


# In[ ]:


train_df.info()
print("----------------------------")
test_df.info()


# Drop original column

# In[ ]:


train_df = train_df.drop(['Name','Ticket','Fare_Range','Cabin'], axis=1)
test_df = test_df.drop(['Name','Ticket','Cabin'], axis=1)


# In[ ]:


corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# Drop the attributes I think unhelpful

# In[ ]:


#train_df = train_df.drop(['Fare','S','SibSp','Parch'], axis=1)
#test_df = test_df.drop(['Fare','S','SibSp','Parch'], axis=1)


# In[ ]:


X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# In[ ]:


X_train.info()
print("----------------------------")
X_test.info()


# In[ ]:


test_df = test_df.drop(['Parch_9'], axis=1)


# In[ ]:


X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1).copy()


# Try different model

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
logreg.score(X_train, Y_train)


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred1 = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)


# In[ ]:


K_Neighbors = KNeighborsClassifier()

K_Neighbors.fit(X_train, Y_train)

Y_pred2 = K_Neighbors.predict(X_test)

K_Neighbors.score(X_train, Y_train)


# The place I output the result

# In[ ]:


from sklearn import metrics
model=GaussianNB()
model.fit(X_train,Y_train)
Y_pred3=model.predict(X_test)
model.score(X_train, Y_train)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)
Y_pred4=model.predict(X_test)
model.score(X_train, Y_train)


# In[ ]:


def save_results(predictions, filename):
    with open(filename, 'w') as f:
        f.write("PassengerId,Survived\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (i + 1, pred))


# In[ ]:


save_results(Y_pred,'1.csv' )
save_results(Y_pred1,'2.csv' )
save_results(Y_pred2,'3.csv' )
save_results(Y_pred3,'4.csv' )
save_results(Y_pred4,'5.csv' )


# In[ ]:




