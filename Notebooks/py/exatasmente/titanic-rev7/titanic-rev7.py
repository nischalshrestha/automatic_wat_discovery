#!/usr/bin/env python
# coding: utf-8

# Titanic Data

# In[ ]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


test_df = pd.read_csv("../input/test.csv")
train_df = pd.read_csv("../input/train.csv")


# In[ ]:


train_df.head(1)


# In[ ]:


train_df.drop(['Name','PassengerId','Cabin','Ticket','Fare'],axis=1,inplace=True)
test_df.drop(['Name','Cabin','Ticket','Fare'],axis=1,inplace=True)
#train_df = train_df.dropna()
#test_df = test_df.dropna()
train_df.count()


# In[ ]:


print(train_df['Age'].mean())
print(train_df['SibSp'].mean())
print(train_df['Parch'].mean())


# In[ ]:


a = train_df[['PassengerId','Age','SibSp', 'Parch','Sex','Pclass']]
a = a[a['Age'].isnull()]
a[a['SibSp']==8]



# In[ ]:



media_idade   = train_df["Age"].mean()
idade_std       = train_df["Age"].std()
qt_nan_idade = train_df["Age"].isnull().sum()

#GERAR UMA MÉDIA DE IDADES
rand_1 = np.random.randint(media_idade - idade_std, media_idade + idade_std, size = qt_nan_idade)
rand_1


# In[ ]:


rand_1.mean()


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('IDADE ANTES DE SER CONVERTIDA')
axis2.set_title('IDADE DEPOIS DE SER CONVERTIDA')

media_idade   = train_df["Age"].mean()
idade_std       = train_df["Age"].std()
qt_nan_idade = train_df["Age"].isnull().sum()

#GERAR UMA MÉDIA DE IDADES
rand_1 = np.random.randint(media_idade - idade_std, media_idade + idade+std, size = qt_nan_idade)
train_df["Age"][np.isnan(train_df["Age"])] = rand_1

average_age1   = test_df["Age"].mean()
std_age1       = test_df["Age"].std()
count_nan_age1 = test_df["Age"].isnull().sum()

#GERAR UMA MÉDIA DE IDADES
rand_2 = np.random.randint(average_age1 - std_age1, average_age1 + std_age1, size = count_nan_age1)

test_df["Age"][np.isnan(test_df["Age"])] = rand_2


train_df['Age'].hist(bins=70, ax=axis1)





train_df['Age'].hist(bins=70, ax=axis1)

# CONVERTENDO FLOAT PARA INT
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age']  = test_df['Age'].astype(int)



train_df['Age'].hist(bins=70, ax=axis2)


print(train_df['Age'].mean())
print(train_df['Age'].min())
print(train_df['Age'].max())


# In[ ]:


train_df.describe()


# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


graph = sns.FacetGrid(train_df, col='Survived')
graph.map(plt.hist, 'Age', bins=20)


# In[ ]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[ ]:


train_df = train_df.dropna()


# In[ ]:


full_data = [train_df,test_df]


# In[ ]:


full_data[1].info()


# In[ ]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[ ]:


full_data = [train_df, test_df]
for dataset in full_data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# In[ ]:


for dataset in full_data:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 58), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 58, 'Age']
train_df.head()


# In[ ]:


train_df
print(train_df.columns)
print(test_df.columns)


# In[ ]:





# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
print(train_df.columns)
print(test_df.columns)


# In[ ]:


random_forest = RandomForestClassifier(n_estimators=400)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:


test_df.count()


# In[ ]:




