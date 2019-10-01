#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings 

warnings.filterwarnings(action = 'ignore')
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')


# In[ ]:


df = pd.read_csv('../input/train.csv')
train = pd.read_csv('../input/test.csv')


# In[ ]:


print ('Rows and Columns : ',df.shape)
print ('NaN Values : ',df.isna().sum().values.sum())
print ('Features :',df.columns)
df.head()


# In[ ]:


df = df.drop(columns=['PassengerId','Name','Ticket'])
pid = train.PassengerId
train  = train.drop(columns=['Name','Ticket'])


# In[ ]:


df = df.drop(columns=['Cabin'])
train = train.drop(columns=['Cabin'])
df.head()


# In[ ]:


df.isna().sum()


# We still have null values in Age column and Embarked Columns
# 
# 
# we can deal with Age columns, Embarked we'll see later.
# 
# ### Dealing With missing values

# In[ ]:


from sklearn.preprocessing import Imputer
imp = Imputer(strategy='most_frequent')


# In[ ]:


imp.fit(df.Age.values.reshape(-1,1))
df.Age = imp.transform(df.Age.values.reshape(-1,1))
train.Age = imp.transform(train.Age.values.reshape(-1,1))


# In[ ]:


df = df.dropna()
df.isna().sum(),train.isna().sum()


# Now we can see we have no missing values in Aeg column.
# 
# ### Separating Categorical and Numerical Columns

# In[ ]:


cat_cols = [i for i in df.columns if len(df[i].unique()) < 8]
num_cols = [i for i in df.columns if i not in cat_cols]
cat_cols,num_cols,len(cat_cols),len(num_cols)


# ## Ploting Data

# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.countplot(df.Pclass,hue=df.Survived)


# 
# The graph above shows that passengers with Class have lower chance of surviving comparing to second and first class, and first class passengers have higher chance of surviving.

# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.countplot(df.Sex,hue=df.Survived)


# So there is higher chance of survival if your are woman 😂.
# 
# let's add a child as third category and check for survival chances.

# In[ ]:


df.Sex[df.Age < 16] = 'child'
fig = plt.figure(figsize=(10,10))
sns.countplot(df.Sex,hue=df.Survived)


# Now you have higher chance of survival if you are a woman or a child.
# 
# why does men have to die always 😒😒.
# 
# Let's check Relation between Pclass and Sex.

# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.countplot(df.Pclass,hue=df.Sex)


# So this is why men are having less chance of survivng.
# 
# Let's make a graph for Embarked From to check if there's a relationship between Survival and your City.

# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.countplot(df.Embarked,hue=df.Survived)


# So you have higher chances of dying in you are starting your journey from S.
# 
# Let's see Relation between Station , Sex and Pclass

# In[ ]:


fig,axs = plt.subplots(1,2,figsize=(20,10))
sns.countplot(df.Embarked,hue=df.Sex,ax = axs[0])
sns.countplot(df.Embarked,hue=df.Pclass,ax = axs[1])


# The graphs above shows that There were more people from S comparing to C and Q. That makes sense why they're dying more. Plus more of men embarked from S and they were in Third class. 
# 
# So that makes clear if you are from S,Male and in third Class you'll die 😗😗😗. Now deal with it.
# 
# Lets make Graphs for Parch and SibSp to check if that can be Factor of survival.

# In[ ]:


fig,axs = plt.subplots(1,2,figsize=(20,10))
sns.countplot(df.SibSp,hue=df.Survived,ax = axs[0])
sns.countplot(df.Parch,hue=df.Survived,ax = axs[1])


# Now those are some interesting graphs.
# 
# first graph shows that the less Siblings you have the less chance is there that you'll survive, But there's an outlier at Value 1 in both graphs. That shows that if you have 1 Sibling or 1 Parent or both you have a higher chance of survial.
# 
# Let's scatter them to see if there's a common relation.

# In[ ]:


sns.scatterplot(df.Parch,df.SibSp,hue=df.Survived)


# There's nothing new to derive from this graph. Let's move to the Numerical Part.
# 

# In[ ]:


fig = plt.figure(figsize=(12,10))
sns.distplot(df.Age)


# This distribution shows that there were more people in age group of 20-40.
# 
# Let's check their survival chance.

# In[ ]:


## Making a Age Group Column to easy some things.

df['AgeGroup'] = df.Age
min(df.Age),max(df.Age)


# In[ ]:


## Now lets categorise them

def retGroup(x):
    if x <= 20:
        return "0-20"
    elif 20 < x <= 40:
        return "20-40"
    elif 40 < x  <= 60:
        return "40-60"
    else:
        return "60-80"
    
    
df.AgeGroup = df.AgeGroup.apply(retGroup)


# Now Let's plot it.

# In[ ]:


fig = plt.figure(figsize=(10,10))
sns.countplot(df.AgeGroup,hue=df.Survived)


# This clearly shows that there's no proper trend in age distribution. But there's slight variaion in age group of 0-20 thats because of they're children and probably they were rescued first.

# ### Let's begin with Model Making

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[ ]:


df = df.dropna()
x = df.drop(['Survived','AgeGroup'],axis=1)
y = df.Survived


# In[ ]:





# In[ ]:


## Label Encoding

encSex = LabelEncoder()
encSex.fit(df.Sex.values.reshape(-1,1))
x.Sex = encSex.transform(df.Sex.values.reshape(-1,1))
train.Sex = encSex.transform(train.Sex.values.reshape(-1,1))


# In[ ]:


encEmb = LabelEncoder()
encEmb.fit(df.Embarked.values.reshape(-1,1))
x.Embarked = encEmb.transform(df.Embarked.values.reshape(-1,1))
train.Embarked = encEmb.transform(train.Embarked.values.reshape(-1,1))


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


## Train test spliting

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.shape,y_train.shape


# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gaussian)


# In[ ]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
acc_logreg = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_logreg)


# In[ ]:


# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
acc_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_svc)


# In[ ]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_test)
acc_linear_svc = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_linear_svc)


# In[ ]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_test)
acc_decisiontree = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_decisiontree)


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
acc_randomforest = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_randomforest)


# In[ ]:


# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_knn)


# In[ ]:


# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_test)
acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)
print(acc_gbk)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes',  'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian ,acc_linear_svc, acc_decisiontree,
               acc_gbk]})

models.sort_values(by='Score', ascending=False)


# In[ ]:


models = models.sort_values(by='Score', ascending=False)
fig = plt.figure(figsize=(10,10))
sns.barplot('Score','Model',data=models)


# In[ ]:


train.isna().sum()


# In[ ]:


impFare = Imputer()


# In[ ]:


train.Fare = impFare.fit_transform(train.Fare.values.reshape((-1,1)))


# In[ ]:


pid = train.PassengerId

predictions = gbk.predict(train.drop(columns=['PassengerId']))

output = pd.DataFrame({ 'PassengerId' : pid, 'Survived': predictions })
output.to_csv('submission.csv', index=False)

