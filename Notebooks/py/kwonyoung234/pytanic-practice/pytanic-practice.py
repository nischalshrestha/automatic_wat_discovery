#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing train, test csv set
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combine = pd.concat([train.drop('Survived',1),test])


# In[ ]:


#check the Dataframe
train.head()


# In[ ]:


#check describe of Dataframe
train.describe()


# In[ ]:


#check missing values
print(train.isnull().sum())
print(test.info())


# In[ ]:


#checking the ratio of survivor in titanic
surv = train[train.Survived == 1]
nosurv = train[train.Survived == 0]
surv_col = 'blue'
nosurv_col = 'red'

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"
      %(len(surv), 1.*len(surv)/len(train)*100.0,len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train)))


# In[ ]:


#comparing value with survived (train dataset)

warnings.filterwarnings(action='ignore')
plt.figure(figsize=[12,10]) #or plt.subplots(figsize=(12,10))
plt.subplot(331)
sns.distplot(surv['Age'].dropna().values,bins=range(0,81,1),kde=False, color= surv_col)
sns.distplot(nosurv['Age'].dropna().values,bins=range(0,81,1),kde=False, color= nosurv_col, axlabel='Age')
plt.subplot(332)
sns.barplot(x='Sex', y='Survived', data=train)
plt.subplot(333)
sns.barplot(x='Pclass', y='Survived', data=train)
plt.subplot(334)
sns.barplot(x='Embarked', y='Survived', data=train)
plt.subplot(335)
sns.barplot(x='SibSp', y='Survived', data=train)
plt.subplot(336)
sns.barplot(x='Parch', y='Survived', data=train)
plt.subplot(337)
sns.distplot(np.log10(surv.Fare.dropna().values+1),kde=False,color=surv_col)
sns.distplot(np.log10(nosurv.Fare.dropna().values+1),kde=False,color=nosurv_col,axlabel='Fare')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
#print median age of survivor and non-survivor
print("Median Age of survivor is %.1f, Median Age of Non-survivor is %.1f" %(surv.Age.dropna().median(),nosurv.Age.dropna().median()))
# surv.Age.dropna().median can be converted to np.median(surv.Age.dropna())


# In[ ]:


tab = pd.crosstab(train.SibSp,train.Survived)
print(tab)

stats.binom_test(x=5,n=5,p=0.62)


# In[ ]:


# calculate the known cabin number of total number of cabin
print('There are %i number of known cabin of %i total cabin in train set'
     %(len(train.Cabin.dropna()),len(train)))
print('There are %i number of known cabin of %i total cabin in test set'
     %(len(test.Cabin.dropna()),len(test)))


# In[ ]:


#getting unique number of ticket
print('The unique number of tickets are %i of %i(total number of tickets)'
      %(train.Ticket.nunique(),len(train)))


# In[ ]:


grouped = train.groupby('Ticket')
k = 0
for name, group in grouped:
    if (len(grouped.get_group(name)) > 1):
        print(group.loc[:,['Survived','Name', 'Fare']])
        k += 1
    if (k>10):
        break


# In[ ]:


#Describe overall relation btw columns via heatmap
plt.figure(figsize=[12,10])
sns.heatmap(train.drop('PassengerId',1).corr(),vmax=0.6,square=True,annot=True)


# In[ ]:


# Excepting for non-numeric values, Looking at the visualized graph using pairplot
# cols are Survived,Pclass,Age,SibSp,Parch,Fare
cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
g = sns.pairplot(data=train.dropna(), vars=cols, size=1.5,
                 hue='Survived', palette=[nosurv_col,surv_col])
g.set(xticklabels=[])


# In[ ]:


#showing two plots comparing btw female surv and nosurv and male surv and nosurv by age
fsurv = train[np.logical_and(train.Sex == 'female',train.Survived == 1)]
fnosurv = train[np.logical_and(train.Sex == 'female',train.Survived == 0)]
msurv = train[np.logical_and(train.Sex == 'male', train.Survived == 1)]
mnosurv = train[np.logical_and(train.Sex == 'male', train.Survived == 0)]

plt.figure(figsize=[12,10])
plt.subplot(121)
sns.distplot(fsurv['Age'].dropna().values,bins=range(0,81,1),kde=False,color=surv_col)
sns.distplot(fnosurv['Age'].dropna().values,bins=range(0,81,1),kde=False,color=nosurv_col,axlabel='Female Age')
plt.subplot(122)
sns.distplot(msurv['Age'].dropna().values,bins=range(0,81,1),kde=False,color=surv_col)
sns.distplot(mnosurv['Age'].dropna().values,bins=range(0,81,1),kde=False,color=nosurv_col,axlabel="Male Age")


# In[ ]:


#According to pairplot I gonna find relation btw pclass and Age using violin plot
sns.violinplot(x="Pclass",y="Age",hue="Survived",data=train,split=True)
plt.hlines([0,10],xmin=-1,xmax=3, linestyles='dotted')


# In[ ]:


#Pclass and Sex by Embarked places
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", col="Embarked",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


#the number of people who were on the classes by the Embarked places
tab = pd.crosstab(combine['Embarked'], combine['Pclass'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Port embarked')
dummy = plt.ylabel('Percentage')


# In[ ]:


#Survived ratio of above plot
sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=train)


# In[ ]:


#people who were distributed by sex are divided by Embarked places
tab = pd.crosstab(combine['Embarked'], combine['Sex'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Port embarked')
dummy = plt.ylabel('Percentage')


# In[ ]:


tab = pd.crosstab(combine['Pclass'], combine['Sex'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Pclass')
dummy = plt.ylabel('Percentage')


# In[ ]:


sib = pd.crosstab(train['SibSp'], train['Sex'])
print(sib)
dummy = sib.div(sib.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Siblings')
dummy = plt.ylabel('Percentage')

parch = pd.crosstab(train['Parch'], train['Sex'])
print(parch)
dummy = parch.div(parch.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Parent/Children')
dummy = plt.ylabel('Percentage')


# In[ ]:


sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True)
plt.hlines([0,10], xmin=-1, xmax=3, linestyles="dotted")


# In[ ]:


plt.figure(figsize=[12,10])
plt.subplot(311)
ax1 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==1].dropna().values+1), kde=False, color=surv_col)
ax1 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==1].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax1.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplot(312)
ax2 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==2].dropna().values+1), kde=False, color=surv_col)
ax2 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==2].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax2.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplot(313)
ax3 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==3].dropna().values+1), kde=False, color=surv_col)
ax3 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==3].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax3.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)


# In[ ]:


ax = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train);
ax.set_yscale('log')


# In[ ]:


print(train[train['Embarked'].isnull()])


# In[ ]:


combine.where((combine['Embarked'] !='Q') & (combine['Pclass'] < 1.5) &     (combine['Sex'] == "female")).groupby(['Embarked','Pclass','Sex','Parch','SibSp']).size()


# In[ ]:


train['Embarked'].iloc[61] = "C"
train['Embarked'].iloc[829] = "C"


# In[ ]:


print(test[test['Fare'].isnull()])


# In[ ]:


test['Fare'].iloc[152] = combine['Fare'][combine['Pclass'] == 3].dropna().median()
print(test['Fare'].iloc[152])


# In[ ]:


combine = pd.concat([train.drop('Survived',1),test])
survived = train['Survived']

combine['Child'] = combine['Age']<=10
combine['Cabin_known'] = combine['Cabin'].isnull() == False
combine['Age_known'] = combine['Age'].isnull() == False
combine['Family'] = combine['SibSp'] + combine['Parch']
combine['Alone']  = (combine['SibSp'] + combine['Parch']) == 0
combine['Large_Family'] = (combine['SibSp']>2) | (combine['Parch']>3)
combine['Deck'] = combine['Cabin'].str[0]
combine['Deck'] = combine['Deck'].fillna(value='U')
combine['Ttype'] = combine['Ticket'].str[0]
combine['Title'] = combine['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
combine['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine['Fare'] + 1))).astype('int')
combine['Bad_ticket'] = combine['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
combine['Young'] = (combine['Age']<=30) | (combine['Title'].isin(['Master','Miss','Mlle']))
combine['Shared_ticket'] = np.where(combine.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
combine['Ticket_group'] = combine.groupby('Ticket')['Name'].transform('count')
combine['Fare_eff'] = combine['Fare']/combine['Ticket_group']
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']>16.0, 2, 1)
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']<8.5,0,combine['Fare_eff_cat'])
test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]


# In[ ]:


g = sns.factorplot(x="Sex", y="Survived", hue="Child", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)
tab = pd.crosstab(train['Child'], train['Pclass'])
print(tab)
tab = pd.crosstab(train['Child'], train['Sex'])
print(tab)


# In[ ]:


cab = pd.crosstab(train['Cabin_known'], train['Survived'])
print(cab)
dummy = cab.div(cab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Cabin known')
dummy = plt.ylabel('Percentage')


# In[ ]:


g = sns.factorplot(x="Sex", y="Survived", hue="Cabin_known", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


tab = pd.crosstab(train['Deck'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Deck')
dummy = plt.ylabel('Percentage')


# In[ ]:


stats.binom_test(x=12,n=12+35,p=24/(24.+35.))


# In[ ]:


g = sns.factorplot(x="Deck", y="Survived", hue="Sex", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


print(train['Ttype'].unique())
print(test['Ttype'].unique())


# In[ ]:


tab = pd.crosstab(train['Ttype'], train['Survived'])
print(tab)
sns.barplot(x="Ttype", y="Survived", data=train, ci=95.0, color="blue")


# In[ ]:


tab = pd.crosstab(train['Bad_ticket'], train['Survived'])
print(tab)
g = sns.factorplot(x="Bad_ticket", y="Survived", hue="Sex", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


tab = pd.crosstab(train['Deck'], train['Bad_ticket'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Deck')
dummy = plt.ylabel('Percentage')


# In[ ]:


tab = pd.crosstab(train['Age_known'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Age known')
dummy = plt.ylabel('Percentage')


# In[ ]:


stats.binom_test(x=424,n=424+290,p=125/(125.+52.))


# In[ ]:


g = sns.factorplot(x="Sex", y="Age_known", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


tab = pd.crosstab(train['Family'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Family members')
dummy = plt.ylabel('Percentage')


# In[ ]:


tab = pd.crosstab(train['Alone'], train['Survived'])
print(tab)
sns.barplot('Alone', 'Survived', data=train)


# In[ ]:


g = sns.factorplot(x="Sex", y="Alone", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


tab = pd.crosstab(train['Large_Family'], train['Survived'])
print(tab)
sns.barplot('Large_Family', 'Survived', data=train)


# In[ ]:


g = sns.factorplot(x="Sex", y="Large_Family", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


tab = pd.crosstab(train['Shared_ticket'], train['Survived'])
print(tab)
sns.barplot('Shared_ticket', 'Survived', data=train)


# In[ ]:


tab = pd.crosstab(train['Shared_ticket'], train['Sex'])
print(tab)
g = sns.factorplot(x="Sex", y="Shared_ticket", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


print(combine['Age'].groupby(combine['Title']).count())
print(combine['Age'].groupby(combine['Title']).mean())

print("There are %i unique titles in total."%(len(combine['Title'].unique())))


# In[ ]:


dummy = combine[combine['Title'].isin(['Mr','Miss','Mrs','Master'])]
foo = dummy['Age'].hist(by=dummy['Title'], bins=np.arange(0,81,1))


# In[ ]:


tab = pd.crosstab(train['Young'], train['Survived'])
print(tab)
sns.barplot('Young', 'Survived', data=train)


# In[ ]:


tab = pd.crosstab(train['Young'], train['Pclass'])
print(tab)
g = sns.factorplot(x="Sex", y="Young", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


plt.figure(figsize=[12,10])
plt.subplot(311)
ax1 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==1].dropna().values+1), kde=False, color=surv_col)
ax1 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==1].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax1.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values+1)))
plt.subplot(312)
ax2 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==2].dropna().values+1), kde=False, color=surv_col)
ax2 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==2].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax2.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values+1)))
plt.subplot(313)
ax3 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==3].dropna().values+1), kde=False, color=surv_col)
ax3 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==3].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax3.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values+1)))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)


# In[ ]:


tab = pd.crosstab(train['Fare_cat'], train['Survived'])
print(tab)
sns.barplot('Fare_cat', 'Survived', data=train)


# In[ ]:


g = sns.factorplot(x="Sex", y="Fare_cat", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


combine.groupby('Ticket')['Fare'].transform('std').hist()
np.sum(combine.groupby('Ticket')['Fare'].transform('std') > 0)


# In[ ]:


combine.iloc[np.where(combine.groupby('Ticket')['Fare'].transform('std') > 0)]


# In[ ]:


plt.figure(figsize=[12,10])
plt.subplot(311)
ax1 = sns.distplot(np.log10(surv['Fare_eff'][surv['Pclass']==1].dropna().values+1), kde=False, color=surv_col)
ax1 = sns.distplot(np.log10(nosurv['Fare_eff'][nosurv['Pclass']==1].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax1.set_xlim(0,np.max(np.log10(train['Fare_eff'].dropna().values+1)))
plt.subplot(312)
ax2 = sns.distplot(np.log10(surv['Fare_eff'][surv['Pclass']==2].dropna().values+1), kde=False, color=surv_col)
ax2 = sns.distplot(np.log10(nosurv['Fare_eff'][nosurv['Pclass']==2].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax2.set_xlim(0,np.max(np.log10(train['Fare_eff'].dropna().values+1)))
plt.subplot(313)
ax3 = sns.distplot(np.log10(surv['Fare_eff'][surv['Pclass']==3].dropna().values+1), kde=False, color=surv_col)
ax3 = sns.distplot(np.log10(nosurv['Fare_eff'][nosurv['Pclass']==3].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
ax3.set_xlim(0,np.max(np.log10(train['Fare_eff'].dropna().values+1)))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)


# In[ ]:


print(combine[combine['Fare']>1].groupby('Pclass')['Fare'].std())
print(combine[combine['Fare_eff']>1].groupby('Pclass')['Fare_eff'].std())


# In[ ]:


combine[(combine['Pclass']==1) & (combine['Fare_eff']>0) & (combine['Fare_eff']<10)]


# In[ ]:


combine[(combine['Pclass']==3) & (np.log10(combine['Fare_eff'])>1.2)]


# In[ ]:


ax = sns.boxplot(x="Pclass", y="Fare_eff", hue="Survived", data=train)
ax.set_yscale('log')
ax.hlines([8.5,16],-1,4, linestyles='dashed')


# In[ ]:


tab = pd.crosstab(train['Fare_eff_cat'], train['Survived'])
print(tab)
sns.barplot('Fare_eff_cat', 'Survived', data=train)


# In[ ]:


g = sns.factorplot(x="Sex", y="Fare_eff_cat", hue="Embarked", col="Pclass",
                   data=train, aspect=0.9, size=3.5, ci=95.0)


# In[ ]:


combine = pd.concat([train.drop('Survived',1),test])
survived = train['Survived']

combine["Sex"] = combine["Sex"].astype("category")
combine["Sex"].cat.categories = [0,1]
combine["Sex"] = combine["Sex"].astype("int")
combine["Embarked"] = combine["Embarked"].astype("category")
combine["Embarked"].cat.categories = [0,1,2]
combine["Embarked"] = combine["Embarked"].astype("int")
combine["Deck"] = combine["Deck"].astype("category")
combine["Deck"].cat.categories = [0,1,2,3,4,5,6,7,8]
combine["Deck"] = combine["Deck"].astype("int")

test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

train.loc[:,["Sex","Embarked"]].head()


# In[ ]:


ax = plt.subplots( figsize =( 12 , 10 ) )
foo = sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=1.0, square=True, annot=True)


# In[ ]:


combine.groupby('Ticket')['Name'].transform('count')


# In[ ]:


#splitting test data set for avoiding overfitting
training, testing = train_test_split(train, test_size=0.2,random_state=0)
print('Total sample size %i; training sample size = %i, testing sample size = %i'
     %(train.shape[0],training.shape[0],testing.shape[0]))


# In[ ]:


cols = ['Sex','Pclass','Cabin_known','Large_Family','Parch',
        'SibSp','Young','Alone','Shared_ticket','Child']
tcols = np.append(['Survived'],cols)

df = training.loc[:,tcols].dropna()
X = df.loc[:,cols]
y = np.ravel(df.loc[:,'Survived'])


# In[ ]:


clf_log = LogisticRegression()
clf_log = clf_log.fit(X,y)
score_log = clf_log.score(X,y)
print(score_log)


# In[ ]:


pd.DataFrame(list(zip(X.columns,np.transpose(clf_log.coef_))))


# In[ ]:


#Do same procedures 73,75 for making test_df,x,y
cols = ['Sex','Pclass','Cabin_known','Large_Family','Shared_ticket','Young','Alone','Child']
tcols = np.append(['Survived'],cols)

df = training.loc[:,tcols].dropna()
X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:,tcols].dropna()
X_test = df_test.loc[:,cols]
y_test = np.ravel(df_test.loc[:,'Survived'])


# In[ ]:


clf_log = LogisticRegression()
clf_log = clf_log.fit(X,y)
score_log = cross_val_score(clf_log, X, y,cv=5).mean()
print(score_log)


# In[ ]:


#Perceptron
clf_pctr = Perceptron(
    class_weight = 'balanced'
)
clf_pctr = clf_pctr.fit(X,y)
score_pctr = cross_val_score(clf_pctr,X,y,cv=5).mean()
print(score_pctr)


# In[ ]:


#K Nearest Neighbors
clf_knn = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance'
)
clf_knn = clf_knn.fit(X,y)
score_knn = cross_val_score(clf_knn,X,y,cv=5).mean()
print(score_knn)


# In[ ]:


#Support Vector Machine

clf_svm = svm.SVC(
    class_weight='balanced'
)
clf_svm.fit(X,y)
score_svm = cross_val_score(clf_svm,X,y,cv=5).mean()
print(score_svm)


# In[ ]:


#Naive Bayes
"""
Naive Bayes is a rapid classification method. It uses the famous Bayes Theorem under the 'naive' assumption that all predictor features are independent from each other (and only related to the target variable).

Despite this oversimplification Naive Bayes classifiers are performing well in many cases. In addition, they are fast to compute and only require relatively little data to perform well.
"""

clf_bay = GaussianNB()
clf_bay.fit(X,y)
score_bay = cross_val_score(clf_bay,X,y,cv=5).mean()
print(score_bay)


# In[ ]:


#bagging
bagging = BaggingClassifier(
    KNeighborsClassifier(
        n_neighbors = 2,
        weights='distance'
    ),
    oob_score=True,
    max_samples=0.5,
    max_features=1.0
)
clf_bag = bagging.fit(X,y)
score_bag = clf_bag.oob_score_
print(score_bag)


# In[ ]:


#Desicion Tree
clf_tree = tree.DecisionTreeClassifier(
    class_weight='balanced',
    min_weight_fraction_leaf=0.01
)
clf_tree = clf_tree.fit(X,y)
score_tree = cross_val_score(clf_tree,X,y,cv=5).mean()
print(score_tree)


# In[ ]:


#Random Forest
clf_rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=10
)

clf_rf = clf_rf.fit(X,y)
score_rf = cross_val_score(clf_rf,X,y,cv=5).mean()
print(score_rf)


# In[ ]:


#Extremely Randomised Trees
clt_ext = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    n_estimators=1000,
    max_depth=None,
    min_samples_split=10
)
clt_ext.fit(X,y)
score_ext = cross_val_score(clt_ext,X,y,cv=5).mean()
print(score_ext)


# In[ ]:


#Gradient Boost
import warnings
warnings.filterwarnings

clf_gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.5,
    random_state=0
).fit(X,y)
clf_gb = clf_gb.fit(X,y)
score_gb = cross_val_score(clf_gb,X,y,cv=5).mean()
print(score_gb)


# In[ ]:


#Ada Boost
clf_ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
clf_ada.fit(X,y)
score_ada = cross_val_score(clf_ada,X,y,cv=5).mean()
print(score_ada)


# In[ ]:


#Extreme Gradient Boosting
clf_xgb= xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    subsamples=0.5,
    learning_rate=0.1
)
clf_xgb.fit(X,y)
score_xgb = cross_val_score(clf_xgb,X,y,cv=5).mean()
print(score_xgb)


# In[ ]:


#Light GBM
clf_lgb = lgb.LGBMClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
    )
clf_lgb.fit(X,y)
score_lgb = cross_val_score(clf_lgb, X, y, cv=5).mean()
print(score_lgb)


# In[ ]:


clf_ext = ExtraTreesClassifier(max_features='auto',bootstrap=True,oob_score=True)
param_grid = { "criterion" : ["gini", "entropy"],
              "min_samples_leaf" : [1, 5, 10],
              "min_samples_split" : [8, 10, 12],
              "n_estimators": [20, 50, 100]}

gs = GridSearchCV(estimator=clf_ext, param_grid=param_grid, scoring='accuracy', cv=3)
gs = gs.fit(X,y)
print(gs.best_score_)
print(gs.best_params_) #parameters what gets best performance via gridsearch


# In[ ]:


clf_ext = ExtraTreesClassifier(
    max_features = 'auto',
    bootstrap = True,
    oob_score =True,
    criterion = 'gini',
    min_samples_leaf = 1,
    min_samples_split = 12,
    n_estimators = 20
)
clf_ext = clf_ext.fit(X,y)
score_ext = clf_ext.score(X,y)
print(score_ext)
pd.DataFrame(list(zip(X.columns,np.transpose(clf_ext.feature_importances_)))).sort_values(1,ascending=False)


# In[ ]:


clf = clf_ext
scores = cross_val_score(clf,X,y,cv=5)
print(scores)
print("mean score is = %.3f, Std deviation is = %.3f" %(score.mean(),score.std()))
#=> np.mean(scores) & np.std(scores)


# In[ ]:


score_ext_test = clf_ext.score(X_test,y_test)
print(score_ext_test)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Gradient Boosting', 'Bagging KNN', 
              'Decision Tree','XGBoost','LightGBM','ExtraTree','Perceptron', 'Naive Bayes'],
    'Score': [score_svm, score_knn, score_log, score_rf, score_gb, score_bag,
              score_tree,score_xgb,score_lgb,score_ext,score_pctr, score_bay]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


summary = pd.DataFrame(list(zip(X.columns,     np.transpose(clf_tree.feature_importances_),     np.transpose(clf_rf.feature_importances_),     np.transpose(clf_ext.feature_importances_),     np.transpose(clf_gb.feature_importances_),     np.transpose(clf_ada.feature_importances_),     np.transpose(clf_xgb.feature_importances_),     np.transpose(clf_lgb.feature_importances_),     )), columns=['Feature','Tree','RF','Extra','GB','Ada','XGBoost','LightGBM'])
  
summary['Median'] = summary.median(1)
summary.sort_values('Median', ascending=False)


# In[ ]:


clf_vote = VotingClassifier(
    estimators = [
        ('knn',clf_knn),
        ('svm',clf_svm),
        ('extra',clf_ext),
        ('xgb',clf_xgb),
        ('percep',clf_pctr),
        ('logistics',clf_log),
    ],
    weights=[2,2,3,3,1,2],
    voting='hard'
)
clf_vote.fit(X,y)

scores = cross_val_score(clf_vote,X,y,cv=5,scoring='accuracy')
print('Voting: Accuracy: %0.2f (+/- %0.2f)'%(scores.mean(),scores.std()))


# In[ ]:


train = X

ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS,random_state=SEED)

class SklearnHelper(object):
    def __init__(self,clf,seed=0,params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train,y_train)
        
    def predict(self,x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)


# In[ ]:


# function for out-of-fold prediction
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    # split data in NFOLDS training vs testing samples
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # select train and test sample
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        # train classifier on training sample
        clf.train(x_tr, y_tr)
        
        # predict classifier for testing sample
        oof_train[test_index] = clf.predict(x_te)
        # predict classifier for original test sample
        oof_test_skf[i, :] = clf.predict(x_test)
    
    # take the median of all NFOLD test sample predictions
    # (changed from mean to preserve binary classification)
    oof_test[:] = np.median(oof_test_skf,axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


# function for out-of-fold prediction
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    # split data in NFOLDS training vs testing samples
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        # select train and test sample
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        # train classifier on training sample
        clf.train(x_tr, y_tr)
        
        # predict classifier for testing sample
        oof_train[test_index] = clf.predict(x_te)
        # predict classifier for original test sample
        oof_test_skf[i, :] = clf.predict(x_test)
    
    # take the median of all NFOLD test sample predictions
    # (changed from mean to preserve binary classification)
    oof_test[:] = np.median(oof_test_skf,axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# In[ ]:


# Put in our parameters for selected classifiers
# Random Forest parameters
rf_params = {
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
}

# Extra Trees Parameters
et_params = {
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# In[ ]:


# Create objects for each classifier
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=svm.SVC, seed=SEED, params=svc_params)


# In[ ]:


# Create Numpy arrays of train, test and target dataframes to feed into our models
y_train = y
train = X
foo = test.loc[:,cols]
x_train = train.values 
x_test = foo.values


# In[ ]:


# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")


# In[ ]:


base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'SVM' : svc_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()


# In[ ]:


plt.figure(figsize=(12,10))
foo = sns.heatmap(base_predictions_train.corr(), vmax=1.0, square=True, annot=True)


# In[ ]:


x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)


# In[ ]:


clf_stack = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 scale_pos_weight=1)
clf_stack = clf_stack.fit(x_train, y_train)
stack_pred = clf_stack.predict(x_test)


# In[ ]:


scores = cross_val_score(clf_stack, x_train, y_train, cv=5)
print(scores)
print("Mean score = %.3f, Std deviation = %.3f"%(np.mean(scores),np.std(scores)))


# In[ ]:


clf = clf_vote
df2 = test.loc[:,cols].fillna(method='pad')
surv_pred = clf.predict(df2)


# In[ ]:


submit = pd.DataFrame({'PassengerId' : test.loc[:,'PassengerId'],
                       #'Survived': surv_pred.T})
                       'Survived': stack_pred.T})
submit.to_csv("../working/submit.csv", index=False)
#submit.to_csv("submit.csv", index=False)


# In[ ]:


submit.head()


# In[ ]:


submit.shape

