#!/usr/bin/env python
# coding: utf-8

# **<font size=4>This is my advanced study of the first competition I attended on Kaggle</font>**
# 
# 
# **<font size=2>I believe that everybody is not unfamliar with the Titanic</font>**
# 
# 
# 
# ![picture](http://www.oscars.org/sites/oscars/files/1083_dpk02718_04_t3d-4k-004.jpg)

# As a data scientist, I think we should all be aware of the ["Black Swam"](https://en.wikipedia.org/wiki/Black_swan_theory)
# 
# **Those things that we can hardly predict**
# 
# To make this world better, I think we should work hard to make right decisions based on the analysis and prediction, but meanwhile, be aware of the probability of happening the so-called Black Swam.
# Hope the tragedy like this will never happen again.

# **<font size=5>Now, let's back to the topic</font>**
# 
# 
# **At first, I think this is a great chance to have a thorough look at my foundation of data science skills**
# 
# Therefore, first, I want to have my plan of this case here.
# * Understand Dataset
# * Data Imputation
# * Exploratory Data Analysis (EDA)
# * Features Engineering & Data Munging
# * Modeling
# * Validation

# **<font size=5>Understanding</font>**
# 
# 
# **First, input data and modules and take a look at it**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from string import ascii_letters
import seaborn as sns
plt.style.use('ggplot')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# **<font size=4>To be more precise, we need to understand the meanings of each columns.</font>**
# 
# 
# **Variable Notes**
# 
# **pclass**: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# **sibsp**: The dataset defines family relations in this way...
# 
# *Sibling* = brother, sister, stepbrother, stepsister
# 
# *Spouse* = husband, wife (mistresses and fiancés were ignored)
# 
# **parch**: The dataset defines family relations in this way...
# 
# *Parent* = mother, father
# 
# *Child* = daughter, son, stepdaughter, stepson
# 
# Some children travelled only with a nanny, therefore parch=0 for them.
# 
# **About PassengerId & Ticket, obviously, PassengerId is just the order number of each row, and I think Ticket is just a random string of each ticket.**

# In[ ]:


train_df.drop(['PassengerId','Ticket'],axis=1,inplace=True)
test_df.drop('Ticket',axis=1,inplace=True)


# Now, I want to see how many null values in both data

# In[ ]:


pd.isnull(train_df).sum()


# In[ ]:


pd.isnull(test_df).sum()


# And have a basic understand of how the data distribute

# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# **<font size=5>Data Imputation</font>**

# Here I put train set and test set together so that I can do preprocess at the same time and after data imputation I copy a set of training set so that I can do EDA with it.

# In[ ]:


target = train_df.Survived
train = train_df.drop('Survived',axis=1)
test = test_df
train['is_train'] = 1
test['is_train'] = 0
train_test = pd.concat([train,test],axis=0)


# **A smart way to impute Age !!!** Refer to : https://www.kaggle.com/ash316/eda-to-prediction-dietanic/notebook

# In[ ]:


train_test['Initial']=0
for i in train_test:
    train_test['Initial']=train_test.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(train_test.Initial,train_test.Sex).T.style.background_gradient(cmap='summer_r')


# In[ ]:


train_test['Initial'].replace(['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Master','Mlle','Mme','Ms','Rev','Sir'],
                            ['Other','Other','Other','Mr','Mrs','Other','Other','Miss','Mr','Mr','Miss','Miss','Miss','Other','Mr'],inplace=True)


# In[ ]:


train_test.groupby('Initial')['Age'].mean()


# In[ ]:


train_test.loc[(train_test.Age.isnull())&(train_test.Initial=='Mr'),'Age']=30
train_test.loc[(train_test.Age.isnull())&(train_test.Initial=='Mrs'),'Age']=37
train_test.loc[(train_test.Age.isnull())&(train_test.Initial=='Miss'),'Age']=22
train_test.loc[(train_test.Age.isnull())&(train_test.Initial=='Other'),'Age']=45


# In[ ]:


train_test.Fare.mode()


# In[ ]:


train_test.loc[(train_test.Fare.isnull()),'Fare']=8.05


# In[ ]:


train_test.Embarked.mode()


# In[ ]:


train_test["Embarked"] = train_test["Embarked"].fillna("S")


# In[ ]:


train_test["Cabin"] = train_test["Cabin"].fillna("No")


# Finally, make sure there is no more NA value

# In[ ]:


pd.isnull(train_test).sum()


# Since I dropped the PassengerId of training set but keep that of testing set for later use, it is fine now.

# In[ ]:


df = train_test[train_test.is_train == 1]


# **<font size=5>EDA</font>**

# In[ ]:


df.drop(['PassengerId','is_train','Initial'],axis=1,inplace=True)


# In[ ]:


df['Survived'] = train_df['Survived']


# In[ ]:


df.describe()


# In[ ]:


corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})


# In[ ]:


f,ax=plt.subplots(2,3,figsize=(12,16))
sns.countplot('Pclass',data=df,ax=ax[0,0])
ax[0,0].set_title('Pclass distribution')
sns.countplot('Sex',data=df,ax=ax[0,1])
ax[0,1].set_title('Sex distribution')
sns.countplot('Age',data=df,ax=ax[0,2])
ax[0,2].set_title('Age distribution')
sns.countplot('SibSp',data=df,ax=ax[1,0])
ax[1,0].set_title('SibSp distribution')
sns.countplot('Parch',data=df,ax=ax[1,1])
ax[1,1].set_title('Parch distribution')
sns.countplot('Embarked',data=df,ax=ax[1,2])
ax[1,2].set_title('Embarked distribution')


# This is a more thorough look at how data distribute.
# 
# Here we can see most of the people are in PClass1 and the proportions of male and female are close.
# Age are normal distribution.

# **And now, let's dig deeper with whether these people survived or not**

# In[ ]:


plt.title("Pclass & Survival Distribution")
plt.hist([df[df['Survived']==1]['Pclass'],df[df['Survived']==0]['Pclass']],bins=3,label=['Survived', 'Dead'])
plt.legend()
plt.show()


# In[ ]:


pd.crosstab(df.Pclass,df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')


# About 2 thirds of the people in PClass1 survived and only half of the people in PClass3 survived.
# I'm considering treating PClass as numbers or dummy variable. But now, I'll just leave it.

# In[ ]:


plt.title("Age & Survival Distribution")
plt.hist([df[df['Survived']==1]['Age'],df[df['Survived']==0]['Age']],bins = 10,label=['Survived', 'Dead'])
plt.legend()
plt.show()


# In terms of age, most of the people under 40 survived. And the ratio gets worse by the age gets older.
# 
# Let's separate these people into groups based on age.

# In[ ]:


df['Age_Range']=pd.qcut(df['Age'],7)
df.groupby(['Age_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# Here we can see that people aged between 28 and 30 are less likely to survive.
# 
# Mayber we can separate them into males and females to look deeper.

# In[ ]:


sns.factorplot('Age_Range','Survived',hue='Sex',data=df)


# Obviously, most groups of female are more likely to survive. And boys are more likely to survive than male adults.

# In[ ]:


plt.title("SibSp & Survival Distribution")
plt.hist([df[df['Survived']==1]['SibSp'],df[df['Survived']==0]['SibSp']],bins=10,range=[0,9],label=['Survived', 'Dead'])
plt.legend()
plt.show()


# In[ ]:


pd.crosstab(df.SibSp,df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')


# It is special that people with 1 or 2 SibSp are most likely to survived in terms of SibSp.
# 
# And those with too many SibSp didn't make it survived.

# In[ ]:


plt.title("Parch & Survival Distribution")
plt.hist([df[df['Survived']==1]['Parch'],df[df['Survived']==0]['Parch']],bins=7,range=[0,7],label=['Survived', 'Dead'])
plt.legend()
plt.show()


# In[ ]:


pd.crosstab(df.Parch,df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')


# It's interesting that except people with Parch more than 4, it seems people with more Parch are more likely to survive.

# In[ ]:


plt.title("Fare & Survival Distribution")
plt.hist([df[df['Survived']==1]['Fare'],df[df['Survived']==0]['Fare']],bins=10,label=['Survived', 'Dead'])
plt.legend()
plt.show()


# In[ ]:


df['Fare_Range']=pd.qcut(df['Fare'],12)
df.groupby(['Fare_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# This is interesting to me as well, since that we can see people with fare price between 7.25 and 7.775 are most likely survived. 
# 
# But basically,except that group, it seems the higher the fare the more likely to survive.

# In[ ]:


df['NameLen'] = df.Name.apply(lambda x : len(x))


# In[ ]:


plt.title("Length of name & Survival Distribution")
plt.hist([df[df['Survived']==1]['NameLen'],df[df['Survived']==0]['NameLen']],bins=10,label=['Survived', 'Dead'])
plt.legend()
plt.show()


# In[ ]:


df['NameLen_Range']=pd.qcut(df['NameLen'],12)
df.groupby(['NameLen_Range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# Though this might sounds weird, people with longer name seem to be more likely to survive.
# 
# (Maybe from a rich family)

# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch']


# In[ ]:


df.FamilySize.describe()


# In[ ]:


plt.title("FamilySize & Survival Distribution")
plt.hist([df[df['Survived']==1]['FamilySize'],df[df['Survived']==0]['FamilySize']],range(0,11),label=['Survived', 'Dead'])
plt.legend()
plt.show()


# In[ ]:


pd.crosstab(df.FamilySize,df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')


# Except those with more than 4 family members, basically, the bigger the family size the more likely to survive.

# In[ ]:


sns.countplot('Sex',hue='Survived',data=df)


# In[ ]:


pd.crosstab(df.Sex,df.Survived).apply(lambda r: r/r.sum(), axis=1).style.background_gradient(cmap='summer_r')


# Without a doubt, the proportion of women survived are much higher than that of men

# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=df)


# It seems that females in PClass 3 are less likely to survive compared with the other females

# In[ ]:


sns.factorplot('Fare_Range','Pclass',hue='Survived',data=df)


# This is pretty straightforward.
# 
# Those with higher fare are not more likely to survive.

# In[ ]:


sns.factorplot('Embarked','Survived',hue='Sex',data=df)


# In this case, I'll treat Embarked as a dummy variable

# **<font size=5>Features Engineering & Data Munging</font>**

# In[ ]:


train_test['Fare_Range']=pd.qcut(train_test['Fare'],12)


# In[ ]:


train_test['NameLen'] = train_test.Name.apply(lambda x : len(x))


# In[ ]:


train_test['Sex'] = train_test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train_test.describe()


# In[ ]:


train_test['FamilySize'] = train_test['Parch'] + train_test['SibSp']


# In[ ]:


train_test['Alone']=0
train_test.loc[(train_test.FamilySize==0),'Alone']=1


# In[ ]:


train_test['Low_Parch']=0
train_test.loc[(train_test.Parch<=3),'Low_Parch']=1


# In[ ]:


train_test['Low_SibSp']=0
train_test.loc[(train_test.SibSp<=2),'Low_SibSp']=1


# In[ ]:


train_test['CabinClass'] = train_test['Cabin'].astype(str).str[0]


# In[ ]:


Cabin  = pd.get_dummies(train_test['CabinClass'],prefix='Cabin',drop_first=False)

train_test = pd.concat([train_test,Cabin],axis=1).drop(['Cabin','CabinClass'],axis=1)


# In[ ]:


Fare_Range  = pd.get_dummies(train_test['Fare_Range'],prefix='Fare_Range',drop_first=False)

train_test = pd.concat([train_test,Fare_Range],axis=1).drop('Fare_Range',axis=1)


# In[ ]:


Embarked  = pd.get_dummies(train_test['Embarked'],prefix='Embarked',drop_first=False)

train_test = pd.concat([train_test,Embarked],axis=1).drop('Embarked',axis=1)


# In[ ]:


Initial  = pd.get_dummies(train_test['Initial'],prefix='Initial',drop_first=False)

train_test = pd.concat([train_test,Initial],axis=1).drop('Initial',axis=1)


# In[ ]:


train_test.info()


# In[ ]:


train_test = train_test.drop(['Name'],axis=1)


# **<font size=5>Modeling</font>**

# In[ ]:


train = train_test[train_test.is_train == 1].drop(['PassengerId','is_train'],axis=1)

test = train_test[train_test.is_train == 0].drop(['PassengerId','is_train'],axis=1)


# In[ ]:


train['Survived'] = train_df['Survived']


# In[ ]:


len(train_test)


# In[ ]:


len(train)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(train,train.Survived,test_size=0.3,random_state=3)


# In[ ]:


X_train = X_train.drop('Survived',axis = 1)
X_test = X_test.drop('Survived',axis = 1)


# In[ ]:


model = LogisticRegression()
model.fit(X_train, Y_train)
model.score(X_train,Y_train)


# In[ ]:


model = RandomForestClassifier(n_estimators=30)
model.fit(X_train, Y_train)
model.score(X_train, Y_train)


# In[ ]:


model = KNeighborsClassifier()
model.fit(X_train, Y_train)
model.score(X_train, Y_train)


# In[ ]:


model = RandomForestClassifier(n_estimators=30)
model.fit(X_train, Y_train)
prediction = model.predict(test)
test['PassengerId'] = test_df['PassengerId']
submission = pd.concat([test[['PassengerId']],pd.DataFrame(prediction)],axis=1)
submission.columns = ['PassengerId', 'Survived']
submission.to_csv('submission.csv',index=False)


# **Refer to :**
# 
# https://www.kaggle.com/ash316/eda-to-prediction-dietanic
# https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# https://www.kaggle.com/startupsci/titanic-data-science-solutions
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# 
# Thanks for sharing

# In[ ]:




