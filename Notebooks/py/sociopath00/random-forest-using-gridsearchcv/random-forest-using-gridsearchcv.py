#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Predicton

# The objective is to predict if passanger has survived or not.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

get_ipython().magic(u'matplotlib inline')


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

print("Train dataset has {} samples and {} attributes".format(*train.shape))
print("Test dataset has {} samples and {} attributes".format(*test.shape))


# In[ ]:


train.head()


# We have 11 feature columns and target variable **Survived** which is binary.

# Pclass, Sex and Embarked are **Categorical** Features while Age, SibSp, Parch and Fare are **continuous** variables.

# We will use Name, Ticket and Cabin variable in Feature Engineering

# ## EDA

# In[ ]:


fig , ax = plt.subplots(figsize=(6,4))
sns.countplot(x='Survived', data=train)
plt.title("Count of Survival")
plt.show()


# In[ ]:


n=len(train)
surv_0=len(train[train['Survived']==0])
surv_1=len(train[train['Survived']==1])

print("% of passanger survived in train dataset: ",surv_1*100/n)
print("% of passanger not survived in train dataset: ",surv_0*100/n)


# Passanger not survived has edge over survived passanger.  

# And even if we do nothing we would get approximately 61% accuracy by simple marking all passangers as not survived(**Accuracy Paradox**). So our aim should be to get accuracy higher than this.

# ### Let's find correlation between Numeric Variable

# In[ ]:


cat=['Pclass','Sex','Embarked']
num=['Age','SibSp','Parch','Fare']


# In[ ]:


corr_df=train[num]  #New dataframe to calculate correlation between numeric features
cor= corr_df.corr(method='pearson')
print(cor)


# In[ ]:


fig, ax =plt.subplots(figsize=(8, 6))
plt.title("Correlation Plot")
sns.heatmap(cor, mask=np.zeros_like(cor, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()


# There's no strong correlation between any two variables. The strongest correlation is between **SibSp** and **Parch** features (0.414).

#  I would like to keep all the features as there is no strong evidence of data redundancy.

# ### Let's use chi-square test to understand relationship between categorical variables and target variable

# In[ ]:


csq=chi2_contingency(pd.crosstab(train['Survived'], train['Sex']))
print("P-value: ",csq[1])


# In[ ]:


csq2=chi2_contingency(pd.crosstab(train['Survived'], train['Embarked']))
print("P-value: ",csq2[1])


# In[ ]:


csq3=chi2_contingency(pd.crosstab(train['Survived'], train['Pclass']))
print("P-value: ",csq3[1])


# P values for features Sex, Embarked and Pclass are very low. So we can reject our Null Hypothesis which is these features are independent and have no relationship with target variable

# So these features contribute by providing some information.

# ## Visualization 

# First Let's check the impact of feature **Sex** on **Survived**

# In[ ]:


fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Survived', data=train, hue='Sex')
ax.set_ylim(0,500)
plt.title("Impact of Sex on Survived")
plt.show()


# We can say that Female passangers have higher probability of survival than Male passangers

# In[ ]:


fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Survived', data=train, hue='Embarked')
ax.set_ylim(0,500)
plt.title("Impact of Embarked on Survived")
plt.show()


# Ratio of Survived and Not Survived passangers for S and Q Embarked are similar but Passengers from C embarked have higer chances of survival.

# In[ ]:


fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Survived', data=train, hue='Pclass')
ax.set_ylim(0,400)
plt.title("Impact of Pclass on Survived")
plt.show()


# Passengers from **Pclass 3** have lesser chances of Survival while passengers from **Pclass 1** have higher chances of survival

# In[ ]:


fig, ax=plt.subplots(1,figsize=(8,6))
sns.boxplot(x='Survived',y='Fare', data=train)
ax.set_ylim(0,300)
plt.title("Survived vs Fare")
plt.show()


# Average Fare for passangers who survived is higher than not survived.

# ## Handling Missing Values

# * Let's check which features contain missing values

# In[ ]:


print(train.isnull().sum())


# In[ ]:


print(test.isnull().sum())


# Only 4 features have missing values

# #### Age

# In[ ]:


train['Age'].describe()


# Let's replace missing values by median of Age.

# In[ ]:


med=np.nanmedian(train['Age'])
train['Age']=train['Age'].fillna(med)
test['Age']=test['Age'].fillna(med)


# #### Cabin

# In[ ]:


train['Cabin'].value_counts()


# Let's replace NaN by 0

# In[ ]:


train['Cabin']=train['Cabin'].fillna(0)
test['Cabin']=test['Cabin'].fillna(0)


# #### Embarked****

# In[ ]:


train['Embarked'].value_counts()


# Let's replace the NaN by mode

# In[ ]:


train['Cabin']=train['Cabin'].fillna("S")


# #### Fare

# In[ ]:


train['Fare'].describe()


# In[ ]:


med=np.nanmedian(train['Fare'])
test['Fare']=test['Fare'].fillna(med)


# ## Feature Engineering

# from **cabin** let's create a new feature **hasCabin** 

# In[ ]:


train['hasCabin']=train['Cabin'].apply(lambda x: 0 if x==0 else 1)
test['hasCabin']=test['Cabin'].apply(lambda x: 0 if x==0 else 1)


# Let's combine SibSp and Parch features to create new one **FamilyMem**

# In[ ]:


train['FamilyMem']=train.apply(lambda x: x['SibSp']+x['Parch'], axis=1)
test['FamilyMem']=test.apply(lambda x: x['SibSp']+x['Parch'], axis=1)


# Let's use prefixes in the name to Create a new column **Title** 

# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


# In[ ]:


train['title']=train['Name'].apply(get_title)
test['title']=test['Name'].apply(get_title)


# In[ ]:


title_lev1=list(train['title'].value_counts().reset_index()['index'])
title_lev2=list(test['title'].value_counts().reset_index()['index'])


# In[ ]:


title_lev=list(set().union(title_lev1, title_lev2))
print(title_lev)


# ### Assigning datatypes

# In[ ]:


train['title']=pd.Categorical(train['title'], categories=title_lev)
test['title']=pd.Categorical(test['title'], categories=title_lev)


# In[ ]:


cols=['Pclass','Sex','Embarked','hasCabin','title']
fcol=['Pclass','Sex','Embarked','hasCabin','title','Age','FamilyMem','Fare']


# In[ ]:


for c in cols:
    train[c]=train[c].astype('category')
    test[c]=test[c].astype('category')


# ###  Let's create dummy variables

# In[ ]:


train_df=train[fcol]
test_df=test[fcol]


# In[ ]:


train_df=pd.get_dummies(train_df, columns=cols, drop_first=True)
test_df=pd.get_dummies(test_df, columns=cols, drop_first=True)


# In[ ]:


y=train['Survived']


# ## Model

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(train_df, y, test_size=0.3, random_state=42)


# ### Random Forest

# In[ ]:


rfc=RandomForestClassifier(random_state=42)


# In[ ]:


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)


# In[ ]:


CV_rfc.best_params_


# In[ ]:


rfc1=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=8, criterion='gini')


# In[ ]:


rfc1.fit(x_train, y_train)


# In[ ]:


pred=rfc1.predict(x_test)


# In[ ]:


print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))


# In[ ]:


op_rf=rfc1.predict(test_df)


# In[ ]:


op=pd.DataFrame(test['PassengerId'])
op['Survived']=op_rf


# In[ ]:


op.to_csv("op_rf.csv", index=False)

