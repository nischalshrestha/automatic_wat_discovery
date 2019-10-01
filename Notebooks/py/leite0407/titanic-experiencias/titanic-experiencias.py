#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train_df.head()


# # Análise dos Dados #
# Fazer uns gráficos bonitinhos, só para tentarmos "perceber" os dados.

# In[ ]:


# Set multiple plots distribution
figs, axs = plt.pyplot.subplots(nrows = 3, ncols = 3, figsize=(30,15))

sns.barplot(x="Pclass", y="Survived", data=train_df, ax = axs[0][0])
sns.barplot(x="Sex", y="Survived", data=train_df, ax = axs[0][1])
sns.barplot(x="Parch", y="Survived", data=train_df, ax = axs[0][2])
sns.barplot(x="SibSp", y="Survived", data=train_df, ax = axs[1][0])
sns.boxplot(x="Survived", y="Fare", data=train_df[(train_df["Sex"] == "female") & (train_df["Fare"] < 100)], ax = axs[1][1])
sns.scatterplot(x="Age", y="Fare", hue="Survived", data=train_df[(train_df["Fare"] < 60) & (train_df["Age"] < 60)], ax = axs[1][2])
sns.barplot(x="Embarked", y="Survived", data=train_df, ax=axs[2][1])


# Agora podiamos fazer mais gráficos, com outros fatores. Por exemplo, fazer gráficos só para homens, ou só para adultos. Se tiver paciência acrescento-os depois. 

# # Feature Creation
# Vamos criar:
# * Deck - Extraída de Cabin. A primeira letra representa o Deck em que ficava a Cabin.
# * AgeIsNaN - Exactamente o que diz. Indica se Age é ou não NaN.

# In[ ]:


def cab_to_deck(cab):
    if type(cab) is float or cab[0] == 'T':
        return "N"
    else:
        return cab[0]
    
train_df["Deck"] = train_df["Cabin"].apply(cab_to_deck)
test["Deck"] = test["Cabin"].apply(cab_to_deck)


# In[ ]:


cabin_is_nan = train_df["Cabin"].isna().sum() / len(train_df["Cabin"])
print("Percentage of NaN: ", cabin_is_nan*100)

figs, axs = plt.pyplot.subplots(ncols = 2, figsize = (30,5))

sns.barplot(x="Deck", y="Survived", data=train_df, ax = axs[0])
sns.countplot(x="Deck", data=train_df, ax = axs[1])


# Ou seja, não temos dados para a Cabin, e consequentemente para o Deck, em 77% dos dados. Mas parece o facto de sabermos ou não em que Deck alguem estava é bastante relevante.

# In[ ]:


# Criar a Feature
train_df["AgeIsNaN"] = train_df["Age"].isna()
test["AgeIsNaN"] = test["Age"].isna()


# In[ ]:


sns.barplot(x="AgeIsNaN", y="Survived", data=train_df)


# In[ ]:


# Substituir valores NaN em Age
train_df["Age"].fillna(train_df["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)


# # Processamento dos Dados #
# Agora vamos só tentar processar um bocado os dados, de modo a poderem ser usados melhores pelos modelos. 
# * Vamos aplicar feature scaling a "Fare" e "Age".
# * One Hot Encoding a Pclass, Sex, SibSp, Parch, Deck e Embarked. 
# * Vamos retirar Cabin e Ticket

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[ ]:


ids = test["PassengerId"]
test = test.drop(columns = ["PassengerId"])


# In[ ]:


# Estabelecer um máximo para Fare, 150, para ter melhores resultados ao usar MinMaxScaler
train_df.loc[train_df["Fare"] > 150, "Fare"] = train_df.loc[train_df["Fare"] > 150, "Fare"].apply(lambda x: 100)
test.loc[test["Fare"] > 150, "Fare"] = test.loc[train_df["Fare"] > 150, "Fare"].apply(lambda x: 100)

test["Fare"].fillna(test["Fare"].mean(), inplace = True)

sns.distplot(train_df["Fare"])


# In[ ]:


scaler = MinMaxScaler()
train_df[["Age", "Fare"]] = scaler.fit_transform(train_df[["Age", "Fare"]])
test[["Age", "Fare"]] = scaler.fit_transform(test[["Age", "Fare"]])


# In[ ]:


train_df = train_df.drop(columns = ['Ticket', 'Cabin', 'Name'])
test = test.drop(columns = ['Ticket', 'Cabin', 'Name'])


# In[ ]:


# Converter as features em dummies
pclass_dummies = pd.get_dummies(train_df["Pclass"], prefix='pclass')
sex_dummies = pd.get_dummies(train_df["Sex"], prefix='sex')
#sibsp_dummies = pd.get_dummies(train_df["SibSp"], prefix='sibsp')
#parch_dummies = pd.get_dummies(train_df["Parch"], prefix='parch')
deck_dummies = pd.get_dummies(train_df["Deck"], prefix='deck')
embarked_dummies = pd.get_dummies(train_df["Embarked"], prefix='embarked')

train_df = train_df.join([pclass_dummies, sex_dummies, deck_dummies, embarked_dummies])

pclass_dummies = pd.get_dummies(test["Pclass"], prefix='pclass')
sex_dummies = pd.get_dummies(test["Sex"], prefix='sex')
#sibsp_dummies = pd.get_dummies(test["SibSp"], prefix='sibsp')
#parch_dummies = pd.get_dummies(test["Parch"], prefix='parch')
deck_dummies = pd.get_dummies(test["Deck"], prefix='deck')
embarked_dummies = pd.get_dummies(test["Embarked"], prefix='embarked')

test = test.join([pclass_dummies, sex_dummies, deck_dummies, embarked_dummies])


# In[ ]:


# Apagar antigas features convertidas, Ticket, Cabin e Name
train_df = train_df.drop(columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Deck', 'Embarked'])
test = test.drop(columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Deck', 'Embarked'])


# In[ ]:


# Separar as labels dos dados
y = train_df["Survived"]
X = train_df.drop(columns = ["Survived", "PassengerId"])


# In[ ]:


# Separar os dados em casos de teste e de treino
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


# # Model making
# Agora que já temos todas as features prontas, podemos começar a criar Models. 
# Vamos experimentar:
# * SVM
# * Logistic Regression
# * Nearest Neighbours
# * Decision Tree
# * Random Forest
# * Gradient Boosted Decision Trees
# 
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.model_selection import GridSearchCV, cross_val_score


# ### Logistic Regression

# In[ ]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[ ]:


print(logreg.score(X_train, y_train))
print(cross_val_score(logreg, X, y, cv = 10).mean())


# In[ ]:


parameters = {'C' : range(1, 20)}

cv = GridSearchCV(logreg, parameters, cv = 10)

cv.fit(X, y)

print(cv.best_score_)


# ### Nearest Neighbours

# In[ ]:


neighbours = KNeighborsClassifier()
neighbours.fit(X_train,y_train)


# In[ ]:


print(neighbours.score(X_train, y_train))
print(cross_val_score(neighbours, X, y).mean())


# In[ ]:


parameters = {'n_neighbors' : range(1, 35)}

cv = GridSearchCV(neighbours, parameters)

cv.fit(X, y)

print(cv.best_score_)
print(cv.best_params_)


# In[ ]:


print(cv.best_estimator_.score(X, y))
print(cross_val_score(cv.best_estimator_, X, y, cv=10).mean())


# In[ ]:


predictions = cv.best_estimator_.predict(test)

submission = pd.DataFrame({'PassengerId' : ids, 'Survived' : predictions})

submission.to_csv('Neighbors.csv', index=False)


# In[ ]:


submission


# ### SVM

# In[ ]:


svc = svm.SVC(C = 100)
svc.fit(X_train, y_train)


# In[ ]:


print(svc.score(X_train, y_train))
print(cross_val_score(svc, X, y).mean())


# In[ ]:


parameters = {'C' : range(1, 100, 5), 'kernel' : ['rbf', 'poly'], 'degree' : [2, 3, 5, 7]}

cv = GridSearchCV(svc, parameters)

cv.fit(X, y)

print(cv.best_score_)
print(cv.best_params_)


# In[ ]:


print(cv.best_estimator_.score(X, y))
print(cross_val_score(cv.best_estimator_, X, y, cv=10).mean())


# In[ ]:




