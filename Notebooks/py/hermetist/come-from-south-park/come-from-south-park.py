#!/usr/bin/env python
# coding: utf-8

# A test on Titanic

# In[ ]:


import pandas as pd
import numpy as nm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model  import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

titanic_train = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')


# In[ ]:



# Take a look at data 
# Only showed number types
print(titanic_train.describe())

print('-------------------')
print(titanic_train.isnull().sum())
print('-------------------')
print(titanic_test.isnull().sum())
print('-------------------')
titanic_train.head()

# info is used to the types of data


# In[ ]:


titanic_train.info()
print('------------------------')
titanic_test.info()


# In[ ]:


print(titanic_train["Embarked"].unique())
print(titanic_test["Embarked"].unique())
print(titanic_train["SibSp"].unique())
print(titanic_test["SibSp"].unique())
print(titanic_train["Pclass"].unique())
print(titanic_test["Pclass"].unique())


# In[ ]:


print(titanic_train.columns.values)
print(titanic_test.columns.values)
#There is a question: if there are some data elements missed in test. 
#Should I completed it and then predict? or just drop all we needn't element types?


# In[ ]:


from numpy import arange
#sns.set(font_scale=1)

#pd.options.display.mpl_style = 'default'
titanic_train.hist(bins=10,figsize=(9,7),grid=False)


#print(len(x))
#sns.lmplot(x, y = 'PassengerId', data = titanic_train)
#for d
#print(titanic["PassengerId"])


# In[ ]:


#x = arange(1,titanic_train.shape[0] + 1, 1)
#print(titanic_train.columns.values)
#plt.plot(x, titanic_train["PassengerId"])
# I just found that PassengerId is userless......


# In[ ]:


g = sns.FacetGrid(titanic_train, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple")


# In[ ]:


g = sns.FacetGrid(titanic_train, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age", edgecolor="w").add_legend()


# In[ ]:


g = sns.FacetGrid(titanic_train, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["v", "^"]), size = 4.5)
#hue_kws seems useless
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender , Age and Fare')


# In[ ]:


titanic_train.Embarked.value_counts().plot(kind='bar', alpha=0.55)
#This plot comes from pandas's plot.
plt.title("Passengers per boarding location")


# In[ ]:


g = sns.kdeplot(titanic_train["Age"], shade=True, color="r")
#print(titanic_train["Age"].unique())     # Age contains NaN


# In[ ]:


#g = sns.FacetGrid(titanic_train, hue = "Survived", col = "Sex", margin_titles=True)
#g.map(sns.plt.ked, "Age")
g = sns.FacetGrid(titanic_train, row = "Survived", col = "Sex", margin_titles = True)
g.map(sns.kdeplot, "Age", shade=True)
# Is there a bug or anything? when I didn't print anything. The g.map will plot the same 2 things.
print(titanic_train["Survived"].unique())
print(titanic_train["Sex"].unique())
#g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()


# In[ ]:


sns.factorplot(x = 'Embarked',y="Survived", data = titanic_train,color="b")
# To view the factor.


# In[ ]:


sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=titanic_train, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    #.set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Passenger Class')


# In[ ]:


ax = sns.boxplot(x="Survived", y="Age", 
                data=titanic_train)
ax = sns.stripplot(x="Survived", y="Age",
                   data=titanic_train, jitter=True,
                   edgecolor="gray")
sns.plt.title("Survival by Age",fontsize=12)


# In[ ]:


corr=titanic_train.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='YlGnBu')
plt.title('Correlation between features')


# In[ ]:


g = sns.factorplot(x="Fare", y="Embarked",
                    hue="Sex", row="Pclass",
                    data=titanic_train[titanic_train.Embarked.notnull()],
                    orient="h", size=2, aspect=3.5, 
                   palette={'male':"purple", 'female':"blue"},
                    kind="violin", split=True, cut=0, bw=.2)
#print(titanic_train.head())
# So far, in fact, only two


# The reason that we have to fill all missed data up. Because, some algorithms need full matrix-like dataset, in fact, a lot.  
# 
# How much effect the the filled data have? How to quantify this kind of effect?
# 
# Tree-structure algorithms perhaps perform better in such missed data.

# In[ ]:


titanic_train[titanic_train['Embarked'].isnull()]


# In[ ]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data = titanic_train)
ax = sns.stripplot(x="Embarked", y="Fare", hue ="Pclass",
                   data = titanic_train, jitter=True,
                   edgecolor="gray")


# In[ ]:


sns.boxplot(x="Embarked", y="Age", hue="Sex", data=titanic_train)
ax = sns.stripplot(x="Embarked", y="Age", hue ="Sex",
                   data=titanic_train, jitter=True,
                   edgecolor="gray")


# In[ ]:


titanic_train["Embarked"] = titanic_train["Embarked"].fillna('C')
# either S or C. The only question is I don't thikd just Fare and Pclass and Sex can be
# used to  determine the Embarked. In fact it's more make sense thar use Cabin and Ticket do that.
# I do it later.


# There is a problem. In fact, all information just lay down there. But it's me to extract features. Is there any possible that can extract feature automatic? 
# 

# In[ ]:


titanic_test[titanic_test['Fare'].isnull()]

