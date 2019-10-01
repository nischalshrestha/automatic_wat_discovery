#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importando bibliotecas
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:





# In[ ]:


#obtendo os dados
data = pd.read_csv("../input/train.csv")
data_test_final = pd.read_csv("../input/test.csv")


# In[ ]:


#separação dos conjuntos de treinamento e teste
train_set, test_set = train_test_split(data, 
                                       test_size = 0.2, 
                                       random_state = 42)

train_set_predictors = train_set.drop('Survived',axis=1)
train_set_labels = train_set['Survived'].copy()
test_set_predictors = test_set.drop('Survived',axis=1)
test_set_labels = test_set['Survived'].copy()


# In[ ]:


train_set_predictors.shape


# In[ ]:


test_set_predictors.shape


# In[ ]:


#Exploração dos dados
train_set.head()


# In[ ]:


#dados estatísticos
train_set.describe()


# In[ ]:


#Histogramas
train_set.hist(bins=50, figsize=(20,15))
plt.show()


# In[ ]:


# Avaliação dos dados e pre-processamento

#PassengerID, Ticket, Name e Cabin foram ignorados

# Atributos numéricos e catecóricos
numeric_atrib = [ 'Age', 'SibSp', 'Parch', 'Fare']
cat_atrib = ['Sex', 'Embarked', 'Pclass']


# In[ ]:


# Atributos numéricos
numeric_data = train_set_predictors[numeric_atrib]
numeric_data.head()


# In[ ]:


numeric_data.info()
#Age apresenta valores faltando


# In[ ]:


# Substitui valores faltando pela média
imputer = Imputer(strategy='median')
imputer.fit(numeric_data)
numeric_data_complete = imputer.transform(numeric_data)


# In[ ]:


# Normalização
scaler = StandardScaler()
scaler.fit(numeric_data_complete)
numeric_data_normalized = scaler.transform(numeric_data_complete)
numeric_data_normalized


# In[ ]:


# Atributos Categóricos
categoric_data = train_set_predictors[cat_atrib]
categoric_data.head()


# In[ ]:


# Valores faltando
categoric_data.info()
# 2 passageiros não tem informação de "Embark"


# In[ ]:


null_columns=categoric_data.columns[categoric_data.isnull().any()]
categoric_data[categoric_data.isnull().any(axis=1)][null_columns].head()


# In[ ]:


categoric_data["Pclass"] = categoric_data["Pclass"].astype('str')


# In[ ]:


# Dummy variables 
categoric_data_encoded = pd.get_dummies(categoric_data)
categoric_data_encoded.head()


# In[ ]:


# Não há necessidade de remover as linhas com valores faltando
categoric_data_encoded.loc[829]


# In[ ]:


# Combinando variaveis numéricas e categoricas

full_data = np.concatenate((categoric_data_encoded.values, numeric_data_normalized),1 )


# In[ ]:


# Algoritmo de AM

# Suport Vector Machine
svc_classifier = SVC()


# In[ ]:


#Grid Search

# Grid de parâmetros a serem testados
param_grid = [
        {'kernel': ['linear'], 'C': [ 0.1, 0.3, 1, 3 ,10]},
        {'kernel': ['rbf'], 'C': [0.1, 0.3, 1, 3, 10],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
         ]

grid_search = GridSearchCV(svc_classifier, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(full_data, train_set_labels)


# In[ ]:


#Resultados do Grid Search
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
    


# In[ ]:


# Melhor resultado
best_accuracy = grid_search.best_score_
best_accuracy


# In[ ]:


best_parameters = grid_search.best_params_
best_parameters


# In[ ]:


#% melhor modelo
final_model = grid_search.best_estimator_
final_model


# In[ ]:


# Previsão do grupo de teste
def prepare_data (data, numeric_atrib, cat_atrib, scaler = None):

    # Atributos numéricos
    numeric_data = data[numeric_atrib]
    
    # Valores faltando
    imputer = Imputer(strategy="median")
    imputer.fit(numeric_data)
    numeric_data_complete = imputer.transform(numeric_data)

    if scaler == None:
        # Normalização
        scaler = StandardScaler()
        scaler.fit(numeric_data_complete)
        
    numeric_data_normalized = scaler.transform(numeric_data_complete)
    
    # Atributos Categóricos
    categoric_data = data[cat_atrib]

    # Dummy variables 
    categoric_data_encoded = pd.get_dummies(categoric_data).values
    # Combinando variaveis numéricas e categoricas
    full_data = np.concatenate((categoric_data_encoded, numeric_data_normalized),1 )
    
    return full_data, scaler

def titanic_train(full_data, labels):

    svc_classifier = SVC()
    
    param_grid = [
        {'kernel': ['linear'], 'C': [0.3, 1, 3 ,10, 30]},
        {'kernel': ['rbf'], 'C': [0.3, 1, 3, 10, 30],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
         ]

    grid_search = GridSearchCV(svc_classifier, param_grid, cv=5, scoring='accuracy', verbose=0)
    grid_search.fit(full_data, labels)
    
    print(grid_search.best_score_)
    
    final_model = grid_search.best_estimator_
    
    return final_model


train_predictors = train_set_predictors
train_labels = train_set_labels
train_full_data, train_scaler = prepare_data(train_predictors, numeric_atrib, cat_atrib)

final_model = titanic_train(train_full_data, train_labels)

test_full_data, _ = prepare_data(test_set_predictors, numeric_atrib, cat_atrib, train_scaler)

final_predictions = final_model.predict(test_full_data)
test_accuracy = accuracy_score(final_predictions, test_set_labels)
test_accuracy


# In[ ]:


final_model


# In[ ]:


#%% Previsão para competição

train_predictors = data.drop('Survived',axis=1)
train_labels = data['Survived'].copy()
train_full_data, train_scaler = prepare_data(train_predictors, numeric_atrib, cat_atrib)

final_model = titanic_train(train_full_data, train_labels)

test_predictors = data_test_final
PassengerId = test_predictors['PassengerId']

test_full_data, _ = prepare_data(test_predictors, numeric_atrib, cat_atrib, train_scaler)

final_predictions = final_model.predict(test_full_data)

StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': final_predictions })

    
StackingSubmission.to_csv("StackingSubmission_2018.2.csv", index=False)

