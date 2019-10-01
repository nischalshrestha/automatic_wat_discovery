#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# ### Загружаем данные

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
data_for_ID = pd.read_csv('../input/test.csv')


# ### Посмотрим на данные

# In[ ]:


train_data.head()


# ### Избавимся от, как кажется, признаков, которые не несут особой информации

# In[ ]:


train_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test_data.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)


# In[ ]:


all_data = pd.concat([test_data, train_data])


# ### Проведем EDA

# In[ ]:


# Посмотрим как Pclass влияет на шанс пассажира спастись
print(train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean())
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train_data)


# ### Таким образом, мы видим, что класс, которым плыл пассажир, по-разному влияет на его шанс спастись

# In[ ]:


### Так же посмотрим как пол влияет на шанс пассажира спастись
print(train_data[['Sex', 'Survived']].groupby(['Sex']).mean())
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train_data)


# ### Мы видим, что большинство спасшихся пассажиров - женщины

# In[ ]:


# Посмотрим как заплаченные деньги влияют на шанс пассажира спастись
g = sns.FacetGrid(train_data, col='Survived')
g = g.map(sns.distplot, "Fare")


# ### Чем больше ты заплатил, тем больше твой шанс спастись

# In[ ]:


# Посмотрим как возраст влияет на шанс пассажира спастись
g = sns.FacetGrid(train_data, col='Survived')
g = g.map(sns.distplot, "Age")


# ### Скорее всего первых на спасительные шлюпки сажали женщин и детей

# In[ ]:


# Посмотрим, как количесво родственников влияет на шанс пассажира спастись
sns.catplot(x='SibSp', y='Survived', data=train_data, kind='bar')


# ### Можно заметить, что чем меньше родственников было у пассажира, тем больше был его шанс спастись

# ### Заполним пропуски в данных и добавим новые признаки

# In[ ]:


def munge_data(data):
    #Замена пропусков на медианы
    data["Age"] = data.apply(lambda r: data.groupby("Sex")["Age"].median()[r["Sex"]] 
                                      if pd.isnull(r["Age"]) else r["Age"], axis=1)
    #Замена пропусков на медианы
    data["Fare"] = data.apply(lambda r: all_data.groupby("Pclass")["Fare"].median()[r["Pclass"]] 
                              if pd.isnull(r["Fare"]) else r["Fare"], axis=1)
    # Gender - замена
    genders = {"male": 1, "female": 0}
    data["Sex"] = data["Sex"].apply(lambda s: genders.get(s))
    # Gender - расширение
    gender_dummies = pd.get_dummies(data["Sex"], prefix="SexD", dummy_na=False)
    data = pd.concat([data, gender_dummies], axis=1)
    # Embarkment - замена
    embarkments = {"U": 0, "S": 1, "C": 2, "Q": 3}
    data["Embarked"] = data["Embarked"].fillna("U").apply(lambda e: embarkments.get(e))
    # Embarkment - расширение
    embarkment_dummies = pd.get_dummies(data["Embarked"], prefix="EmbarkedD", dummy_na=False)
    data = pd.concat([data, embarkment_dummies], axis=1)
    # Количество родственников на борту
    data["Relatives"] = data["Parch"] + data["SibSp"]
    
    return(data)


# In[ ]:


train_data_munged = munge_data(train_data).drop(['EmbarkedD_0'],axis=1)
test_data_munged = munge_data(test_data)


# ### Кросс-валидация на три фолда

# In[ ]:


cv = StratifiedKFold(train_data["Survived"], n_folds=3, shuffle=True, random_state=1)


# ### Посмотрим на два алгоритма - Random Forest и K Nearest Neighbors

# In[ ]:


alg = RandomForestClassifier(random_state=1, n_estimators=350, min_samples_split=6, min_samples_leaf=2)
scores = cross_val_score(alg, train_data_munged, train_data_munged["Survived"], cv=cv)
print("Accuracy (random forest): {}".format(scores.mean()))


# In[ ]:


alg_ngbh = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(alg_ngbh, train_data_munged, train_data_munged["Survived"], cv=cv)
print("Accuracy (k-neighbors): {}".format(scores.mean()))


# ### Для решения задачи будем использовать RandomForestClassifier

# In[ ]:


alg.fit(train_data_munged.drop(["Survived"],axis=1), train_data_munged["Survived"])

predictions = alg.predict(test_data_munged)

submission = pd.DataFrame({
    "PassengerId": data_for_ID["PassengerId"],
    "Survived": predictions
})

submission.to_csv("titanic-submission.csv", index=False)

