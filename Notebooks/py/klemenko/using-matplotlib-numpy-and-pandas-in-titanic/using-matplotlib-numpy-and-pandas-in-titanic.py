#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# In this example I want to show the short worksheet and workflow in Machine Learning. This is a  short example how to clear code, prepare for machine learning and how to use most common libraries in Pyton like Pandas, numpy or matplotlib. This is my first try in the Kaggle competition and Machine Learning. If you find any mistakes or wrong definition please inform me and I will correct this in the article. In below article  I based on other articles about Titanic in the Kaggle competition so you can find couple similarities.

# ## Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#Sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.naive_bayes import  MultinomialNB, BernoulliNB,GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import linear_model

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To fill NA values
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, MICE
from sklearn.preprocessing import Imputer


# # Basic Pandas functions
# Pandas give as couple function to check data before analize. 

# ## Import Data
# 
# Pandas has couples function to import data. One is read_csv, others are  **pd.read_excel(xlsx, 'Sheet1')**.  The return value is **data_frame**

# In[ ]:


# Load the data, we set that index_col is the first column, therefore there will be standard index start from 0 for each data.
train_df = pd.read_csv('../input/train.csv', header=0,index_col=0)
test_df = pd.read_csv('../input/test.csv', header=0,index_col=0)


# In[ ]:


full = pd.concat([train_df , test_df]) # concatenate two dataframes


# In[ ]:


SURV = 891
full.info()   # info about dataframe


# In[ ]:


full.head()       # give first couple rows of data


# In[ ]:


full.describe()  # give statistics about data


# In[ ]:


full[:10] # Like in regular Python you can get to the Item by Index


# In[ ]:


full[SURV:899:2] # Like in regular Python you can get to the Item by Index


# In[ ]:


full[(full['Age'] > 5.0) & (full['Age'] < 7.0 ) ] #filter data by columns


# In[ ]:


full[(full['Cabin'].str.contains('B2',na=False)) ] #filter data by columns


# In[ ]:


full.isnull().sum()  # Check with alues are empty


# In[ ]:


#Missing values in the plot
import missingno as msno
msno.matrix(full)


# In[ ]:


msno.dendrogram(full)


# In[ ]:


train_df.groupby(['Pclass','Sex'])['Survived'].sum() # grouping data


# In[ ]:


full['CabinType'] = full['Cabin'].astype(str).str[0]
full['_CabinType'] = pd.Categorical(full.CabinType).codes

full['CabinType2'] = full['Cabin'].astype(str).str[0:2];
full['_CabinType2'] = pd.Categorical(full.CabinType2).codes

full[:SURV].groupby(['Pclass','CabinType'])['Survived'].agg(['count','sum','mean'])


# 
# # Correlation map

# In[ ]:


full['_Sex'] = pd.Categorical(full.Sex).codes
full['_Embarked'] = pd.Categorical(full.Embarked).codes


# In[ ]:


cols = ['Age','_Embarked','Fare','Parch','Pclass','_Sex','SibSp','Survived','_CabinType']
full[cols].corr()


# # Fill NaN values
# 
# In previous examples we found that Age and Cabin aren't filled. To use this data we need them to fill with values. We can use mean, median , remove these values or even Machine Learning to fill these values.

# In[ ]:


full[cols].corr()['_Embarked'].sort_values()


# In[ ]:


full[full['Embarked'].isnull()]


# The highest correlation for Embarked is from column **Fare**, **Pclass**, **cabin**.  

# In[ ]:


val = full[  (full['Pclass'] == 1) 
     & (full['CabinType'] == 'B') 
     ][['Fare','Embarked']];
val.groupby(['Embarked' ])['Fare'].agg(['count','min','max','mean','median'])


# In[ ]:


ax = val.boxplot(column='Fare',by='Embarked');
ax.axhline(80,color='red')


# Based on this information we can assume that this 'Embarked' propably is 'S' (Southampton). 

# In[ ]:


full.set_value(62,'Embarked','S');
full.set_value(830,'Embarked','S');


# ### Fill 'Fare' value

# In[ ]:


full[full['Fare'].isnull()]


# In[ ]:


full[cols].corr()['Fare'].abs().sort_values(ascending=False)


# Fare depends on class, embarked and propably Parch. This is because some tickets were propably family tickets that's  cheaper than normal.

# In[ ]:


val = full[  (full['Pclass'] == 3) 
     & (full['Embarked'] == 'S') 
     & (full['Parch'] == 0) 
      ][['Age','Fare']];

val['Fare'].agg(['min','max','count','mean','median'])


# Base on the Age we can asume that 'Fare' is propably 7.92. So we fill with the median value.

# In[ ]:


full.set_value(1044,'Fare', 7.925); # we set average for this values
full.isnull().sum() 


# # Fill Age value
# 
# Based on the Mice inputation we can fill N/A values for Age.

# In[ ]:


cols = ['Age','_Embarked','Fare','Parch','Pclass','_Sex','SibSp','Survived','_CabinType']
full[cols].corr()['Age'].abs().sort_values(ascending=False)


# ### Fill using MICE

# In[ ]:


X = full[['Age']]

mice = MICE(n_imputations=100, impute_type='col',init_fill_method='mean',min_value=0,verbose=False)
# x = mice.complete(X);

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
ximp = imp.fit_transform(X)

# full['_AgeMICE'] = x[:,0]
full['_AgeImputer'] = ximp[:,0]


# ### Fill using LInearRegression

# In[ ]:


# fill using LinearRegression
cols = ['Pclass','_CabinType','SibSp','Fare','Parch'] 

ageData = full[full['Age'].notnull()]
emptyData = full[full['Age'].isnull()]
Y = ageData['Age'] 
X = ageData[cols] # ,'Parch','Embarked']]

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(X,Y)
print('Linear score: ',regr.score(X,Y))

X = emptyData[cols]
Y = regr.predict(X)
# First we need to set index before 

pred =  pd.concat([pd.Series(Y,emptyData.index),ageData['Age']]).sort_index()
full['_AgeLinear'] = pred


# In[ ]:


print(full['Age'].isnull().sum())
# print(full['_AgeMICE'].isnull().sum())
print(full['_AgeImputer'].isnull().sum())
print(full['_AgeLinear'].isnull().sum())

plt.subplot(221); full['Age'].hist(); 
plt.subplot(222); full['_AgeLinear'].hist();
#plt.subplot(223); full['_AgeMICE'].hist();  
plt.subplot(224); full['_AgeImputer'].hist();


# In[ ]:


full[(full['Age'].isnull())][['Age','_AgeImputer','_AgeLinear']][:20]


# ## Add Age Category column
# In the dataset we have a lot of detailed columns but some of them we want only to have part of the information. In the pandas we can provide new columns with new data.
# 
# 

# In[ ]:


# full['_AgeMICER'] = pd.cut(full['_AgeMICE'],[0,9,18,30,40,50,100]) # Add column with range of Age
full['_AgeImputerR'] = pd.cut(full['_AgeImputer'],[0,9,18,30,40,50,100]) # Add column with range of Age
full['_AgeLinearR'] = pd.cut(full['_AgeLinear'],[0,9,18,30,40,50,100]) # Add column with range of Age

# full[:SURV].groupby('_AgeMICER')['Survived'].agg(['count','sum','mean'])


# In[ ]:


full[:SURV].groupby('_AgeImputerR')['Survived'].agg(['count','sum','mean'])


# In[ ]:


full[:SURV].groupby('_AgeLinearR')['Survived'].agg(['count','sum','mean'])


# In[ ]:


full['AgeCategory'] = pd.cut(full['_AgeLinear'],[0,9,18,30,40,50,100], labels=[9,18,30,40,50,100]) # Add column with range of Age
full['_AgeCategory'] = full['AgeCategory'].cat.codes # Add column with range of Age


# ## Fill Cabin column
# 
# When we have age column, we can now fill Cabin. For calculation we don't use the full number of cabin, because it is inpractical and could have bad influance in the result. We just put only first letter with is dock letter and second one with letter and first number. In the end we check with one is better option.
# 
# For this we use Decision Tree Clasification

# In[ ]:


cols = ['_AgeLinear','_Embarked','Fare','Parch','Pclass','_Sex','SibSp','_CabinType']
full[cols].corr()['_CabinType'].abs().sort_values(ascending=False)


# In[ ]:


clas = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)

cols = ['Pclass','Fare','_AgeLinear','_Embarked','_Sex']
data = full[full['Cabin'].notnull()]
emptyData = full[full['Cabin'].isnull()]

X = data[cols]
Y = data['CabinType']

clas.fit(X,Y)
print('Decision score: ',clas.score(X,Y))

X = emptyData[cols]
Y = clas.predict(X)

# First we need to set index before 
pred =  pd.concat([pd.Series(Y,emptyData.index), data['CabinType']]).sort_index()
full['CabinType'] = pred


# In[ ]:


X = data[cols]
Y = data['CabinType2']

clas.fit(X,Y)
print('Decision score 2: ',clas.score(X,Y))

X = emptyData[cols]
Y = clas.predict(X)

# First we need to set index before 
pred =  pd.concat([pd.Series(Y,emptyData.index), data['Cabin']]).sort_index()
full['CabinType2'] = pred


# In[ ]:


full['_CabinType'] = pd.Categorical(full.CabinType).codes
full['_CabinType2'] = pd.Categorical(full.CabinType2).codes
full[(full['Cabin'].isnull())][['Cabin','CabinType','_CabinType','CabinType2','_CabinType2']][:20]


# 

# # Normalize Fare column

# In[ ]:


from sklearn import preprocessing
full['_Fare'] = preprocessing.scale(full[['Fare']]) [:,0]


# ## Matplotlib - reports

# In[ ]:


for column in ['Pclass','Sex','SibSp','Parch','Embarked']: 
    fig, axes = plt.subplots(nrows=1, ncols=2)

    (train_df
        .groupby(column)['Survived']
        .agg(['count','sum'])
        ).plot.bar(ax=axes[0])
    (train_df
        .groupby(column)['Survived']
        .mean()
        ).plot.bar(ax=axes[1])


# ## Add data for Tickets

# We can also add data about categories who has the same ticket and check of surviving these people. We can assume that people with the same ticket number ar close together and as you can see it's not the same as sum of column 'SibSp' and 'Parch'.

# In[ ]:


full['TicketCounts'] = full.groupby(['Ticket'])['Ticket'].transform('count')

(full
.groupby('TicketCounts')['Survived']
.agg(['count','sum'])
).plot.bar()
plt.show()
(full
.groupby('TicketCounts')['Survived']
.mean()
).plot.bar()
plt.show()

full[['TicketCounts','Ticket','SibSp','Parch']][:20]


# # Get Titles

# In[ ]:


pat = r",\s([^ .]+)\.?\s+"

full['Title'] =  full['Name'].str.extract(pat,expand=True)[0]
full.groupby('Title')['Title'].count()


# In[ ]:


full[:SURV].groupby('Title')['Survived'].agg(['count','sum','mean'])


# In[ ]:


full.loc[full['Title'].isin(['Mille','Ms','Lady']),'Title'] = 'Miss'
full.loc[full['Title'].isin(['Mme','Sir']),'Title'] = 'Mrs'
full.loc[~full['Title'].isin(['Miss','Master','Mr','Mrs']),'Title'] = 'Other' # NOT IN
full['_Title'] = pd.Categorical(full.Title).codes
full.groupby('Title')['Title'].count()


# In[ ]:


(full
.groupby('Title')['Survived']
.agg(['count','sum'])
).plot.bar()
plt.show()
(full
.groupby('Title')['Survived']
.mean()
).plot.bar()
plt.show()


# # Machine Learning

# First we decide witch column we use in the final learning.

# In[ ]:


cols = ['_AgeCategory','_CabinType','_Embarked','Fare','Parch','Pclass','_Sex','SibSp','TicketCounts','_Title']

colsY = cols + ['Survived']
full[:SURV][colsY].corr()['Survived'].abs().sort_values(ascending=False)


# In[ ]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import  MultinomialNB, BernoulliNB,GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import linear_model
from xgboost import XGBClassifier

Y = full[:SURV]['Survived']
X = full[:SURV][cols] # ,'Parch','Embarked']]


classifiers = {
  'Linear':  linear_model.LogisticRegression(),
  'SGDClassifier': SGDClassifier(),
  'SVC': SVC(class_weight='balanced'),
  'LinearSVC':  LinearSVC(),
  'NuSVC': NuSVC(),
  'GaussianNB': GaussianNB(),
  'BernoulliNB': BernoulliNB(), 
  'XGBoost': XGBClassifier()
 # 'MultinomialNB': MultinomialNB()
} 


Xp = full[SURV:][cols]
score = {}
out  = {}

result = pd.DataFrame({'PassengerID': full[SURV:].index })

for c in classifiers:
    clf = classifiers[c]
    clf.fit(X,Y)
    score[c] = clf.score(X,Y)
    result[c]   = clf.predict(Xp)

score


# We have couple of predictions so we can use function to decide wich one we can use to predict our data.

# In[ ]:


#result['Survived'] = result['SVC'] + result['LinearSVC']
def calculate(s):
    Surv = 0
    a = 0
    for c in classifiers: # I score each classifier by sum of the Score
        Surv += (1.0 if s[c]>0 else -1.0)*score[c]
    return pd.Series({'Survived': (1 if Surv > 0 else 0)})

result['Survived'] = result['NuSVC'].T.astype(int) # .apply(calculate, axis=1)
result


# In[ ]:


#output = pd.DataFrame({'PassengerID': full[SURV:].index , 'Survived': result.T.astype(int)})
#output.to_csv('submission2.csv',index=False)

result[['PassengerID','Survived']].to_csv('submission2.csv',index=False)

print('hurra, we find the result.')


# 
