#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re as re

from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
test_indexes=test_data['PassengerId']


# In[ ]:


def detect_outliers(df,n,features):
    outlier_indexes=[]
    
    for col in features:
        Q1=np.percentile(df[col],25)
        Q3=np.percentile(df[col],75)
        IQR=Q3-Q1
        step=IQR*1.5
        outlier_list_col = df[(df[col] < Q1 - step) | (df[col] > Q3 + step )].index
        outlier_indexes.extend(outlier_list_col)
        
    outlier_indexes=Counter(outlier_indexes)
    multiple_outliers = list( k for k, v in outlier_indexes.items() if v > n )
    
    return multiple_outliers

Outliers_to_drop = detect_outliers(train_data,2,["Age","SibSp","Parch","Fare"])


# In[ ]:


train_data.loc[Outliers_to_drop]
train_data=train_data.drop(Outliers_to_drop,axis=0).reset_index(drop=True)
t


# In[ ]:


full_dataset = pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)
full_dataset = full_dataset.fillna(np.nan)
full_dataset.describe()


# In[ ]:


full_dataset.info()


# In[ ]:


#Drop Unecessary features
full_dataset.drop(['PassengerId'],axis=1,inplace=True)


# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train_data[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# In[ ]:


full_dataset["Fare"] = full_dataset["Fare"].fillna(full_dataset["Fare"].median())


# In[ ]:


full_dataset["Fare"] = full_dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)


# In[ ]:


g = sns.heatmap(full_dataset[["Age",'Fare','FamilySize','IsAlone',"Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)


# In[ ]:


full_dataset["Embarked"] = full_dataset["Embarked"].fillna("S")


# In[ ]:


full_dataset.info()


# In[ ]:


full_dataset["Sex"] = full_dataset["Sex"].map({"male": 0, "female":1})


# In[ ]:


index_NaN_age = list(full_dataset["Age"][full_dataset["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = full_dataset["Age"].median()
    age_pred = full_dataset["Age"][((full_dataset['SibSp'] == full_dataset.iloc[i]["SibSp"]) & (full_dataset['FamilySize'] == full_dataset.iloc[i]["FamilySize"]) & (full_dataset['Pclass'] == full_dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        full_dataset['Age'].iloc[i] = age_pred
    else :
        full_dataset['Age'].iloc[i] = age_med


# In[ ]:


full_dataset.info()


# In[ ]:


full_dataset.Cabin.fillna(value = 'X', inplace = True)

'''Keep only the 1st character where Cabin is alphanumerical.'''
full_dataset.Cabin = full_dataset.Cabin.apply( lambda x : x[0])
display(full_dataset.Cabin.value_counts())

'''After processing, we can visualize the absolute and relative frequency of newly transformed Cabin variable.'''
#absolute_and_relative_freq(merged.Cabin)


# In[ ]:


full_dataset['FamilySize']=full_dataset['SibSp']+full_dataset['Parch']+1
full_dataset["IsAlone"]=0
full_dataset.loc[full_dataset['FamilySize']==1,'IsAlone']=1


# In[ ]:


full_dataset['Title'] = full_dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


full_dataset['Title'].value_counts()


# In[ ]:


full_dataset.Title.replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)

'''Put Dona, Jonkheer, Countess, Sir, Lady, Don in bucket Aristocrat.'''
full_dataset.Title.replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)

'''Finally Replace Mlle and Ms with Miss. And Mme with Mrs.'''
full_dataset.Title.replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)


# In[ ]:


ticket = []
for x in list(full_dataset.Ticket):
    if x.isdigit():
        ticket.append('N')
    else:
        ticket.append(x.replace('.','').replace('/','').strip().split(' ')[0])
        
'''Swap values'''
full_dataset.Ticket = ticket
full_dataset.Ticket = full_dataset.Ticket.apply(lambda x : x[0])
display(full_dataset.Ticket.value_counts())


# In[ ]:





# In[ ]:


train_data[['Title','Survived']].groupby('Title',as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


title_map={'Mr':1,'Rare':2,'Master':3,'Miss':4,'Mrs':5}
full_dataset=[train_data,test_data]
for dataset in full_dataset:
    dataset['Title']=dataset['Title'].map(title_map).astype(int)
    
print(train_data.info())
print(test_data.info())


# In[ ]:


full_dataset.drop(['Name'],axis=1,inplace=True)


# In[ ]:


train_data[['Age','Survived']].groupby('Age',as_index=False).mean()


# In[ ]:


train_data['AgeBand']=pd.cut(train_data['Age'],5)
train_data[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)


# In[ ]:


full_dataset.loc[full_dataset['Age']<=16.336,'Age']=4
full_dataset.loc[(full_dataset['Age']>16.336)&(full_dataset['Age']<=32.252),'Age']=3
full_dataset.loc[(full_dataset['Age']>32.252)&(full_dataset['Age']<=48.168),'Age']=2
full_dataset.loc[(full_dataset['Age']>48.168)&(full_dataset['Age']<=64.084),'Age']=1
full_dataset.loc[full_dataset['Age']>64.084,'Age']=0
full_dataset['Age']=full_dataset['Age'].astype(int)


# In[ ]:


full_dataset['FareBand'] = pd.qcut(full_dataset['Fare'], 4)
full_dataset[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=False)


# In[ ]:


full_dataset.loc[full_dataset['Fare']<=7.91,'Fare']=0
full_dataset.loc[(full_dataset['Fare']>7.91)&(full_dataset['Fare']<=14.454),'Fare']=1
full_dataset.loc[(full_dataset['Fare']>14.454)&(full_dataset['Fare']<=31),'Fare']=2
full_dataset.loc[full_dataset['Fare']>31,'Fare']=3
full_dataset['Fare']=full_dataset['Fare'].astype(int)


# In[ ]:


full_dataset.drop(['FareBand'],axis=1,inplace=True)


# In[ ]:


full_dataset.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'FamilySize', 'Ticket']] = full_dataset.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'FamilySize', 'Ticket']].astype('category')


# In[ ]:


full_dataset.Survived = full_dataset.Survived.dropna().astype('int')


# In[ ]:


full_dataset=pd.get_dummies(full_dataset)


# In[ ]:


full_dataset.info()


# In[ ]:


train_data.info()


# In[ ]:


train_data=full_dataset.iloc[:881,:]
test_data=full_dataset.iloc[881:,:]


# In[ ]:


test_data.info()


# In[ ]:



Y_train=train_data['Survived'].astype('int')
X_train=train_data.drop(['Survived'],axis=1)
X_test=test_data.drop(['Survived'],axis=1)
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(train_data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_indexes,
        "Survived": random_forest.predict(X_test)
    })
submission.to_csv("submission.csv",index=False)


# In[ ]:




