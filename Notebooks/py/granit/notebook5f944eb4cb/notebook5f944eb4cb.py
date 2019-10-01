#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:





# In[ ]:


sns.heatmap(train.isnull())


# In[ ]:


sns.countplot(x="Survived", data=train)


# In[ ]:


sns.countplot(x="Survived", data=train, hue="Sex")


# In[ ]:


sns.countplot(x="Survived", data=train[train.Age < 16], hue="Sex")


# In[ ]:


#sns.distplot(train["Fare"], hue="Survived")


# In[ ]:


sns.distplot(train["Age"].dropna(), bins=40)


# In[ ]:


sns.boxplot(x="Pclass", y = "Age", data = train)


# In[ ]:


ages = {x:int(train[train.Pclass == x].Age.median()) for x in range(1,4)}


# In[ ]:


train["Age"] = [ages[x] for x in train.Pclass] 


# In[ ]:


test["Age"] = [ages[x] for x in test.Pclass] 


# In[ ]:


train.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)
test.drop(["Cabin", "Name", "Ticket"], axis=1, inplace=True)


# In[ ]:


train = train.dropna()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.lda import LDA
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier


# In[ ]:


label_cols = ["Sex", "Embarked"]


# In[ ]:


le = {x: preprocessing.LabelEncoder() for x in label_cols}


# In[ ]:


for x in le:
    train[x] = le[x].fit_transform(train[x])
    test[x] = le[x].transform(test[x])


# In[ ]:


labels = train.Survived
train.drop("Survived", axis = 1, inplace=True)


# In[ ]:


test["Fare"] = test["Fare"].fillna(test["Fare"].median())


# In[ ]:





# In[ ]:


pl = Pipeline([('scale', StandardScaler()), ('lda', LDA()), ('clf', LogisticRegression())])
pl = Pipeline([('scale', StandardScaler()), ('lda', PCA(n_components=4)), ('clf', svm.SVC())])
#pl = RandomForestClassifier()
#pl = XGBClassifier(silent=False)
#pl = GaussianNB()
#pl = QuadraticDiscriminantAnalysis()ponents

pl = svm.SVC()

pl = MLPClassifier(hidden_layer_sizes=(30, ), verbose = True, activation= 'tanh', max_iter=100)


# In[ ]:


pl.fit(train, labels)


# In[ ]:


preds = pl.predict(test)


# In[ ]:


len(test)


# In[ ]:


len(preds)


# In[ ]:


res = pd.DataFrame(preds, columns = ["Survived"], index = test.PassengerId)


# In[ ]:


res.to_csv("submit.csv")


# In[ ]:


train.head()


# In[ ]:


lda = LDA(n_components=2)
pca = PCA(n_components=2)


# In[ ]:


scaler = StandardScaler()
trains = scaler.fit_transform(train)

trans = pca.fit_transform(trains, labels)
dfp = pd.DataFrame(trans, index = train.index)

trans = lda.fit_transform(trains, labels)
df = pd.DataFrame(trans, columns = ["lda"], index = train.index)


# In[ ]:


df["labels"] = labels
dfp["labels"] = labels


# In[ ]:


df.plot(kind='scatter', x='lda', y='labels')


# In[ ]:


dfp.plot(kind='scatter', x='pca1', y='pca2',  c="labels")


# In[ ]:


get_ipython().magic(u'matplotlib inline')

