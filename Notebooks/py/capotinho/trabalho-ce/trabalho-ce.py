#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import statsmodels.api as sm
from pandas.core import datetools

import os
print(os.listdir("../input"))


# In[ ]:


# Leitura dos arquivos de treino e teste
raw_train_data = pd.read_csv('../input/train.csv')
raw_test_data = pd.read_csv('../input/test.csv')

# Dataframe com todos os dados, tanto de treino quanto de teste
raw_data = pd.concat([raw_train_data, raw_test_data], axis=0, sort=False).reset_index().drop(columns=['index'])
raw_data.head(10)


# In[ ]:


print('Shape(linhas, colunas) do dataset de treino: {}'.format(raw_train_data.shape))
print('Shape(linhas, colunas) do dataset de teste: {}'.format(raw_test_data.shape))

print('\n')

print('Colunas do dataset de treino:')
print(raw_train_data.columns)
print('\n')
print('Colunas do dataset de teste:')
print(raw_test_data.columns)


# In[ ]:


raw_data.describe(include='all')


# In[ ]:


raw_data.info()


# In[ ]:


print('Quantidade de valores nulos:')
raw_data.isnull().sum()


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

raw_data[['Age', 'Fare']] = imputer.fit_transform(raw_data[['Age', 'Fare']])


# ## Data Engineering

# In[ ]:


def group_age(x):
    if x >= 0 and x < 18:
        return 1
    if x >= 18 and x < 30:
        return 2
    if x >= 30 and x < 50:
        return 3
    if x >= 50:
        return 4
    
def is_child(x):
    if x == 1:
        return 1
    else:
        return 0
    
def is_adult(x):
    if x == 2:
        return 1
    else:
        return 0

new_col = raw_data['Age'].apply(group_age)
raw_data = raw_data.assign(Group_Age = new_col)

is_child_col = raw_data['Group_Age'].apply(is_child)
raw_data = raw_data.assign(is_child = is_child_col)

raw_data.head(10)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# In[ ]:


"""
Modelagem da DecisionTree
"""
param_dis_tree = {
    "max_depth": np.arange(2,10),
    "max_features": ['auto', 'sqrt', 'log2', None, 'auto'],
    "min_samples_leaf": np.arange(2, 10),
    "criterion": ["gini", "entropy"]
}


def tree_param_selection(X, y, nfolds):
    tree = DecisionTreeClassifier(random_state=0)
    tree_cv = GridSearchCV(tree, param_dis_tree, cv=nfolds)
    tree_cv.fit(X, y)
    print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
    print("Best score is {}".format(tree_cv.best_score_))
    return tree_cv.best_params_

def train_tree_model(X_train, X_test, y_train, y_test):
    # Tunning dos hyperparams
    params = tree_param_selection(X_train, y_train, 5)
    
    # Modelagem e treinamento
    tree = DecisionTreeClassifier(random_state=0, **params)
    tree.fit(X_train, y_train)
    
    # Predição e análise de acurárcia
    y_pred = tree.predict(X_test)
    print('DecisionTreeClassifier: {}'.format(accuracy_score(y_test, y_pred)))
    
    return tree


# In[ ]:


"""
Modelagem de RandomForest
"""
param_random_forest = {
    #"n_estimators": np.arange(2,20),
    "max_depth": np.arange(2,10),
    "max_features": ['auto', 'sqrt', 'log2', None, 'auto'],
    "min_samples_leaf": np.arange(2, 10),
    "criterion": ["gini", "entropy"]
}

def random_forest_param_selection(X, y, nfolds):
    """
    Função para selecionar os melhores hyperparams
    """
    forest = RandomForestClassifier(random_state = 0)
    forest_cv = GridSearchCV(forest, param_random_forest, cv=nfolds)
    forest_cv.fit(X, y)
    print("Tuned Random Forest Parameters: {}".format(forest_cv.best_params_))
    print("Best score is {}".format(forest_cv.best_score_))
    return forest_cv.best_params_

def train_forest_model(X_train, X_test, y_train, y_test):
    # Tunning dos hyperparams
    params = random_forest_param_selection(X_train, y_train, 5)
    
    # Modelagem e treinamento
    forest = RandomForestClassifier(random_state=0, **params)
    forest.fit(X_train, y_train)
    
    # Predição e análise de acurárcia
    y_pred = forest.predict(X_test)
    print('RandomForestClassifier1: {}'.format(accuracy_score(y_test, y_pred)))
    
    y_pred = forest.predict(X_test)
    print('RandomForestClassifier2: {}'.format(accuracy_score(y_test, y_pred)))
    
    return forest


# In[ ]:


split_size = raw_train_data.shape[0]

def get_best_model(data):
    X = data.drop(columns=['Survived', 'PassengerId'])
    y = data['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
    # Modelos de teste
    tree = train_tree_model(X_train, X_test, y_train, y_test)
    print('\n')
    # Random Forest
    forest = train_forest_model(X_train, X_test, y_train, y_test)

    # SVC
    svc = SVC(gamma='auto', random_state=0)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print('SVC: {}'.format(accuracy_score(y_test, y_pred)))    
    
    # Linear SVC
    linear_svc = LinearSVC(random_state=0)
    linear_svc.fit(X_train, y_train)
    y_pred = linear_svc.predict(X_test)
    print('LinearSVC: {}'.format(accuracy_score(y_test, y_pred)))
    
    # XGBClassifier
    xgb_reg = XGBClassifier(random_state=0, n_estimators=100000000, learning_rate=0.05)
    xgb_reg.fit(X_train, y_train,
                verbose=False,
                early_stopping_rounds=20,
                eval_set=[(X_test, y_test)])
    y_pred = xgb_reg.predict(X_test)
    print('XGBClassifier: {}'.format(accuracy_score(y_test, y_pred)))
    
    return tree, forest


# # Primeira versão de dados (somente valores numéricos)
# 

# In[ ]:


numeric_data = raw_data.select_dtypes(exclude=['object'])
numeric_data_columns = numeric_data.columns.tolist()

get_best_model(numeric_data[0:split_size])


# # Segunda versão com valores categóricos

# In[ ]:


cat_num_col = numeric_data_columns + ['Sex', 'Embarked']

categorical_and_numerical_data = raw_data[cat_num_col]
dummy_data = pd.get_dummies(categorical_and_numerical_data, columns=['Sex', 'Embarked'])

tree, forest = get_best_model(dummy_data[0:split_size])
#forest = forest_model(dummy_data[0:split_size])


# In[ ]:


test_data = dummy_data[split_size:].drop(columns=['Survived', 'PassengerId'])

pred = tree.predict(test_data)

submit = pd.DataFrame({'PassengerId': dummy_data[split_size:].PassengerId, 'Survived': pred})
submit['Survived'] = submit['Survived'].astype('int')
submit.to_csv('submission.csv', index=False)

