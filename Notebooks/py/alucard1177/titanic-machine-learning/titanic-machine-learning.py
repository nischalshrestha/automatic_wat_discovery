#!/usr/bin/env python
# coding: utf-8

# # Titanic ML

# ## Загружаем библиотеки работы с данными, а также сами данные

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# ## Проверим, что данные загрузились, заодно посмотрим, как они выглядят

# In[ ]:


train_data.head()


# ## Проверим наличие незаполненных полей

# In[ ]:


train_data.isna().sum()


#    ## Заполним недостающие данные колонки Age медианой остальных значений и пустые ячейки Embarked значением 'S'

# In[ ]:


train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)


# ## Уберем колонки-идентификаторы пассажиров, а также колонку Cabin, т.к. около 80% ее значений не заполнено

# In[ ]:


train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# ## Проведем EDA

# In[ ]:


train_data.describe()


# ### Посмотрим, как пол пассажира повлиял на его выживание

# In[ ]:


#Посмотрим, как пол пассажира повлиял на его выживание
sns.barplot(x='Sex', y='Survived',data=train_data)


# ### Видим, что с корабля спаслось больше женщин, чем мужчин. Скорее всего их сажали на шлюпки в первую очередь.

# In[ ]:


#Посмотрим на распределение по классам пассажиров
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train_data)


# ### Можно заметить, что чем "элитнее" пассажир, тем большая вероятность его выживания.

# In[ ]:


#Посмотрим, как на шансы спастись могло повлиять количество родственников на борту
data = [train_data,test_data]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

sns.barplot(x='relatives', y='Survived', data=train_data)


# ### Из графика видно, что при количестве родственников на борту большем 3, шансы на выживание значительно падают.

# In[ ]:


train_data.head(10)


# ## Закодируем категориальные признаки числами

# In[ ]:


from sklearn.preprocessing import LabelEncoder
def encode_features(data, features):
    for feature in features:
        le = LabelEncoder()
        le.fit(data[feature])
        encoded_column = le.transform(data[feature])
        data[feature] = encoded_column
    return data

to_encode = ['Sex', 'Embarked']
train_data = encode_features(train_data, to_encode)
test_data = encode_features(test_data, to_encode)
train_data.head()


# ## Затем расширим их, добавив бинарные признаки принадлежности к каждой категории

# In[ ]:


sex_binaries = pd.get_dummies(train_data['Sex'], prefix='Sex_')
embark_binaries = pd.get_dummies(train_data['Embarked'], prefix='Embarked_')
train_data = pd.concat([train_data, sex_binaries, embark_binaries], axis=1)

sex_binaries = pd.get_dummies(test_data['Sex'], prefix='Sex_')
embark_binaries = pd.get_dummies(test_data['Embarked'], prefix='Embarked_')
test_data = pd.concat([test_data, sex_binaries, embark_binaries], axis=1)

test_data.head(10)


# In[ ]:


train_data.drop(['Sex', 'SibSp', 'Parch', 'Embarked'], axis=1, inplace=True)
test_data.drop(['Sex', 'SibSp', 'Parch', 'Embarked'], axis=1, inplace=True)

test_data.head(10)


# ## Разбиваем данные на фолды

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']

skf = list(StratifiedKFold(n_splits = 4, shuffle=True, random_state=177).split(X, y))


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
cvs = cross_val_score(tree, X, y, scoring='roc_auc', cv=skf)
print("Score of Decision Tree: {}" .format(cvs.mean()))


# ## KNN (6 neighbors)

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6, n_jobs=12)
cvs = cross_val_score(knn, X, y, scoring='roc_auc', cv=skf)
print("Score of KNN: {}" .format(cvs.mean()))


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=177, n_estimators=100, min_samples_split=6, min_samples_leaf=2)
cvs = cross_val_score(forest, X, y, scoring='roc_auc', cv=skf)
print("Score of Random Forest: {}" .format(cvs.mean()))


# ## Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear')
cvs = cross_val_score(logreg, X, y, scoring='roc_auc', cv=skf)
print("Score of Logistic Regression: {}" .format(cvs.mean()))


# ## Лучший результат показал Random Forest, используем его

# In[ ]:


forest.fit(X, y)
prediction = forest.predict(test_data.drop(['PassengerId'], axis=1))
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": prediction 
    })
submission.to_csv("submission.csv", index=False)


# In[ ]:


submission.head()

