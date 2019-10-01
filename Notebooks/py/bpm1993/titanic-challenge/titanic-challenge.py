#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import lightgbm as lgb


# ### Análise exploratória e tratamento dos dados

# In[ ]:


titanic = pd.read_csv('../input/train.csv')
titanic_test = pd.read_csv('../input/test.csv')
titanic.head()


# Tratamento dos seguintes dados:
# * Separação do prefixo do ticket e do número na coluna Ticket
# * Criação de dummies para sexo
# * Criação de dummies para classe
# * Criação de coluna de título
# * Criação de dummies para Embarked
# * Tratamento das idades faltantes

# In[ ]:


# Criação de funções para separar os valores
def getTicketPrefix(x):
    splitted = x.split(' ')
    if len(splitted) > 1:
        return splitted[0]
    return None

def getTicketNumber(x):
    splitted = x.split(' ')
    if len(splitted) > 1:
        return splitted[1]
    return x


# In[ ]:


# Criação das colunas separadas e a remoção da coluna Ticket
titanic['Ticket Prefix'] = titanic['Ticket'].map(getTicketPrefix)
titanic['Ticket Number'] = titanic['Ticket'].map(getTicketNumber)
titanic_test['Ticket Prefix'] = titanic_test['Ticket'].map(getTicketPrefix)
titanic_test['Ticket Number'] = titanic_test['Ticket'].map(getTicketNumber)
del titanic['Ticket']
del titanic_test['Ticket']
titanic.head()


# In[ ]:


# Criação das dummies para sexo
titanic = pd.concat([titanic, pd.get_dummies(titanic['Sex'], prefix="sex")], axis=1)
titanic_test = pd.concat([titanic_test, pd.get_dummies(titanic_test['Sex'], prefix="sex")], axis=1)
titanic.head()


# In[ ]:


# Criação das dummies para classe
titanic = pd.concat([titanic, pd.get_dummies(titanic['Pclass'], prefix="class")], axis=1)
titanic_test = pd.concat([titanic_test, pd.get_dummies(titanic_test['Pclass'], prefix="class")], axis=1)
titanic.head()


# In[ ]:


# Criação da coluna de título
def getTitle(x):
    search = re.search(r"\,\ (\w+)\.", x)
    if(not isinstance(search, type(None))):
        return search.group(1)
    return ''

titanic['Title']= titanic['Name'].str.extract("\,\ (.+?)\.")
titanic_test['Title']= titanic_test['Name'].str.extract("\,\ (.+?)\.")
titanic.head()


# In[ ]:


# Criação de dummies para os títulos
titanic = pd.concat([titanic, pd.get_dummies(titanic['Title'], prefix="title")], axis=1)
titanic_test = pd.concat([titanic_test, pd.get_dummies(titanic_test['Title'], prefix="title")], axis=1)
titanic.info()


# In[ ]:


# Criação de dummies para Embarked
titanic = pd.concat([titanic, pd.get_dummies(titanic['Embarked'], prefix="embarked")], axis=1)
titanic_test = pd.concat([titanic_test, pd.get_dummies(titanic_test['Embarked'], prefix="embarked")], axis=1)
titanic.info()


# In[ ]:


# Inserção de mediana nas idades faltantes
titanic['Age'][titanic['Age'].isnull()] = titanic['Age'].median()
titanic_test['Age'][titanic_test['Age'].isnull()] = titanic_test['Age'].median()
titanic = pd.concat([titanic, pd.get_dummies(pd.cut(titanic["Age"], [0, 10, 18, 30, 55, np.Inf], labels=["Age - Below 10", "Age - 10-18", "Age - 18-30", "Age - 30-55", "Age - Above 55"]).astype(str))], axis = 1)
titanic.head()


# In[ ]:


# Inserção de mediana nos valores de passagem faltantes
titanic['Fare'][titanic['Fare'].isnull()] = titanic['Fare'].median()
titanic_test['Fare'][titanic_test['Fare'].isnull()] = titanic_test['Fare'].median()


# In[ ]:


# Ordenação das colunas para melhor visualização
titanic = titanic[['Survived', 'Title', 'Name', 'Age', 'Sex', 'sex_female', 'sex_male', 'SibSp', 'Parch', 'Fare', 'Pclass', 'class_1', 'class_2', 'class_3', 'Cabin', 'Embarked', 'Ticket Prefix', 'Ticket Number', 'title_Capt', 'title_Col', 'title_Don', 'title_Dr', 'title_Jonkheer', 'title_Lady', 'title_Major', 'title_Master', 'title_Miss', 'title_Mlle', 'title_Mme', 'title_Mr', 'title_Mrs', 'title_Ms', 'title_Rev', 'title_Sir', 'title_the Countess', 'embarked_C', 'embarked_Q', 'embarked_S', "Age - Below 10", "Age - 10-18", "Age - 18-30", "Age - 30-55", "Age - Above 55"]]
titanic_test = titanic_test[['Title', 'Name', 'Age', 'Sex', 'sex_female', 'sex_male', 'SibSp', 'Parch', 'Fare', 'Pclass', 'class_1', 'class_2', 'class_3', 'Cabin', 'Embarked', 'Ticket Prefix', 'Ticket Number', 'title_Col', 'title_Dr', 'title_Master', 'title_Miss', 'title_Mr', 'title_Mrs', 'title_Ms', 'title_Rev', 'embarked_C', 'embarked_Q', 'embarked_S']]
titanic.head()
titanic_test.head()


# ### Visualização dos dados

# In[ ]:


survived = titanic[titanic['Survived'] == 1]
not_survived = titanic[titanic['Survived'] == 0]
survived.head()
not_survived.head()


# In[ ]:


sns.set(rc={'figure.figsize':(12, 8)})
sns.distplot(survived['Age'].dropna(), hist_kws={"color": "r"}, kde_kws={"color": "r","label": "Survived"})
sns.distplot(not_survived['Age'].dropna(), hist_kws={"color": "g"}, kde_kws={"color": "g","label": "Not Survived"})


# In[ ]:


sns.boxplot(x = 'Survived', y = 'Age', data = titanic)


# In[ ]:


sns.set(rc={'figure.figsize':(12, 8)})
sns.factorplot(x="Title", y="Survived",size=7, aspect=2,  data=titanic, kind='bar', order=['the Countess', 'Mlle', 'Sir', 'Lady', 'Ms', 'Mme', 'Mrs', 'Miss', 'Master', 'Major', 'Col', 'Dr', 'Mr', 'Jonkheer', 'Capt', 'Rev', 'Don'])


# ## Modelos Estatísticos

# ### Tratamento dos dados

# In[ ]:


# Remoção de variáveis categóricas para o modelo
titanic = titanic.drop(columns=['Title', 'Name', 'Sex', 'Cabin', 'Embarked', 'Ticket Prefix', 'Ticket Number'])
titanic.info()


# In[ ]:


# Separando X e y
titanic_backup = titanic
X = titanic.drop('Survived', axis=1)
y = titanic['Survived']


# In[ ]:


# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[ ]:


y_true = titanic.iloc[y_test.index.values]['Survived']


# In[ ]:


# Criação de dataframe para performances
perf = pd.DataFrame()
acc = pd.DataFrame(columns=['Model', 'Accuracy'])

# Função para inserção de dados de performance
def getMetrics(y_test, y_pred, name):
    metrics = pd.DataFrame(list(precision_recall_fscore_support(y_test, y_pred)), index=['Precision', 'Recall', 'F1 Score', 'a'])
    metrics = metrics.drop(metrics.index[3]).transpose()
    metrics['Model'] = name
    metrics['range'] = range(2)
    return metrics


# ### Decision Tree - Gini Impurity

# In[ ]:


# Instanciando modelo
clf = tree.DecisionTreeClassifier(criterion = 'gini')
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'Decision Tree - Gini Impurity'))
acc = acc.append(pd.DataFrame({'Model': 'Decision Tree - Gini Impurity', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# In[ ]:


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ### Decision Tree - Information Entropy

# In[ ]:


# Instanciando modelo
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'Decision Tree - Information Entropy'))
acc = acc.append(pd.DataFrame({'Model': 'Decision Tree - Information Entropy', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# In[ ]:


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# ### Random Forests

# In[ ]:


# Instanciando modelo
clf = RandomForestClassifier()
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'Random Forests'))
acc = acc.append(pd.DataFrame({'Model': 'Random Forests', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### Logistic Regression

# In[ ]:


# Instanciando modelo
clf = LogisticRegression()
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'Logistic Regression'))
acc = acc.append(pd.DataFrame({'Model': 'Logistic Regression', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### Support Vector Machines

# In[ ]:


# Instanciando modelo
clf = svm.SVC(gamma='auto')
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'SVM'))
acc = acc.append(pd.DataFrame({'Model': 'Support Vector Machines', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### Gaussian Naive Bayes

# In[ ]:


# Instanciando modelo
clf = GaussianNB()
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'Gaussian Naive Bayes'))
acc = acc.append(pd.DataFrame({'Model': 'Gaussian Naive Bayes', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### Multinomial Naive Bayes

# In[ ]:


# Instanciando modelo
clf = MultinomialNB()
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'Multinomial Naive Bayes'))
acc = acc.append(pd.DataFrame({'Model': 'Multinomial Naive Bayes', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### Bernoulli Naive Bayes

# In[ ]:


# Instanciando modelo
clf = BernoulliNB()
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'Bernoulli Naive Bayes'))
acc = acc.append(pd.DataFrame({'Model': 'Bernoulli Naive Bayes', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### K-Nearest Neighbors

# In[ ]:


# Instanciando modelo
clf = KNeighborsClassifier(n_neighbors=50)
clf = clf.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = clf.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'K-Nearest Neighbors'))
acc = acc.append(pd.DataFrame({'Model': 'K-Nearest Neighbors', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### XG Boost

# In[ ]:


# Instanciando modelo
xgb = XGBClassifier()
xgb.fit(X_train, y_train)


# In[ ]:


# Testando modelo
y_pred = xgb.predict(X_test)


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'XG Boost'))
acc = acc.append(pd.DataFrame({'Model': 'XG Boost', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# ### LightGBM

# In[ ]:


# Convertendo dados
train_data = lgb.Dataset(X_train, label=y_train)
param = {'num_leaves':31, 'num_trees':50, 'objective':'binary'}


# In[ ]:


bst = lgb.train(param, train_data, 10)


# In[ ]:


y_pred = bst.predict(X_test)
for i in range(0,len(y_pred)):
    if y_pred[i]>=.5:
       y_pred[i]=1
    else:  
       y_pred[i]=0


# In[ ]:


# Validando resultados
perf = perf.append(getMetrics(y_test, y_pred, 'LightGBM'))
acc = acc.append(pd.DataFrame({'Model': 'LightGBM', 'Accuracy': [accuracy_score(y_true, y_pred)]}))


# In[ ]:


perf


# ## Visualização da performance

# ### Precisão

# In[ ]:


sns.factorplot(x="Model", y="Precision", size=7, aspect=3.2,  data=perf, kind='bar', hue='range')


# ### Recall

# In[ ]:


sns.factorplot(x="Model", y="Recall", size=7, aspect=3.2,  data=perf, kind='bar', hue='range')


# In[ ]:


## F1 Score


# In[ ]:


sns.factorplot(x="Model", y="F1 Score", size=7, aspect=3.2,  data=perf, kind='bar', hue='range')


# In[ ]:


# Criação de Dataframe com a performance no Kaggle
perf_kaggle = pd.DataFrame({ 'Model': ['Decision Tree - Gini Impurity', 'Decision Tree - Information Entropy', 'Random Forests', 'Logistic Regression', 'Support Vector Machine', 'Gaussian Naive Bayes', 'Multinomial Naive Bayes', 'Bernoulli Naive Bayes', 'K-Nearest Neighbors', 'XG Boost', 'LightGBM'], 'Accuracy Kaggle' : [0.61722, 0, 0.72727, 0.77033, 0.62679, 0.74641, 0.66985, 0.77033, 0.64593, 0.77511, 0.76076]})


# ### Perfomance no Kaggle

# In[ ]:


acc


# In[ ]:


perf_kaggle


# In[ ]:


perf_kaggle.insert(loc=2, column='Accuracy Score', value=acc.Accuracy.values)


# In[ ]:


perf_kaggle = pd.melt(perf_kaggle, value_vars=['Accuracy Kaggle', 'Accuracy Score'], id_vars=['Model'], var_name = 'Origin', value_name='Accuracy')


# In[ ]:


sns.set(rc={'figure.figsize':(30, 8)})
plt.ylim(ymax=1)
sns.barplot(x="Model", y="Accuracy", hue='Origin', data=perf_kaggle)


# In[ ]:




