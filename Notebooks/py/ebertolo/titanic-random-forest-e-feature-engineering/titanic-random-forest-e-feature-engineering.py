#!/usr/bin/env python
# coding: utf-8

# # Titanic - RandomForest e Features Engineering
# 
# Este notebook cria um modelo baseado no dataset do Titanic e usando um algorítmo Random Forest. 
# Esse notebook faz tratativa de dados faltantes, features categóricas e um pouco de feature engineering.

# Vamos começar importando as bibliotecas básicas que vamos usar.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Próximo passo: carregando os dados a partir dos CSVs disponibilizados no Kaggle. Estamos usando a biblioteca pandas para esse propósito.

# In[ ]:


# Vamos iniciar o notebook importanto o Dataset
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# Podemos observar as primeiras linhas dele.
test_df.head()


# In[ ]:


print(test_df.shape, titanic_df.shape)
titanic_df.head()


# Vamos começar com o básico de tratamento desse dataset. Importante: tudo que fizermos vamos fazer no dataset de treinamento e também no de teste.

# ## Tratando a Idade - Imputation

# Verificamos que o campo de idade possui valores nulos:

# In[ ]:


titanic_df['Age'].isnull().any()


# Teremos que preencher isso de algum jeito. Uma abordagem comum nesses casos é usar uma média ou mediana. Vamos usar aqui a mediana do dataset - mas poderíamos agrupar por sexo, por exemplo. Fica a seu critério e experimentação a melhor estratégia de imputation para essa coluna.

# In[ ]:


age_median = titanic_df['Age'].median()
print(age_median)


# In[ ]:


titanic_df['Age'] = titanic_df['Age'].fillna(age_median)
test_df['Age'] = test_df['Age'].fillna(age_median)
print('Existem nulos?', titanic_df['Age'].isnull().any())


# Ótimo! Um problema a menos. Essa técnica que usamos é chamada de "imputation".

# ## Tratando Gênero - LabelEncoding

# Próximo passo: vamos tratar das features categóricas. Queremos transformar features que são de múltiplas categorias em dados numéricos que podemos trabalhar. O caso mais óbivo é sexo.

# In[ ]:


import seaborn as sns
sns.countplot(titanic_df['Sex']);


# Vamos usar aqui um transformer do sklearn chamado LabelEncoder. Ele transforma a primeira categoria no número 0, a segunda no número 1 e assim por diante.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
sex_encoder = LabelEncoder()
sex_encoder.fit(list(titanic_df['Sex'].values) + list(test_df['Sex'].values))


# In[ ]:


sex_encoder.classes_


# In[ ]:


titanic_df['Sex'] = sex_encoder.transform(titanic_df['Sex'].values)
test_df['Sex'] = sex_encoder.transform(test_df['Sex'].values)


# In[ ]:


sns.countplot(titanic_df['Sex'], order=[1,0]);


# Ok, a feature Sex já está devidamente encodada. Vamos dar mais uma espiada nos dados?

# In[ ]:


titanic_df.head()


# Já temos mais colunas numéricas. Vamos estudar o impacto de adicionar essas colunas no nosso modelo. Vamos usar o nosso modelo anterior.

# In[ ]:


feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare']


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(np.array(titanic_df[feature_names].values), 
                                                      np.array(titanic_df['Survived'].values),
                                                      test_size=0.2,
                                                      random_state=42)
                                                      
                                                      
print(train_X.shape)
print(valid_X.shape)                                           
print(train_y.shape)
print(valid_y.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

#Hiperparametros
rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7)

#Treino
rf_clf.fit(train_X, train_y)

print("Score Treino")
print(rf_clf.score(train_X, train_y))


print("Score Validação")
print(rf_clf.score(valid_X, valid_y))


# In[ ]:


import seaborn as sns

plt.title('Exibindo a importância de cada atributo do dataset')
sns.barplot(rf_clf.feature_importances_, feature_names);


# Agora vaamos incluir algumas features (colunas) )adicionais:

# In[ ]:


seed = 42

feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex']

X = np.array(titanic_df[feature_names].values)
y = np.array(titanic_df['Survived'].values)

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(X,y, test_size=0.2,random_state=seed)
                                                                                                        
#print(train_X.shape)
#print(valid_X.shape)                                           
#print(train_y.shape)
#print(valid_y.shape)

rf_clf = RandomForestClassifier(random_state=seed, n_estimators=200, max_depth=5)
rf_clf.fit(train_X, train_y)

print('Score de treino:',rf_clf.score(train_X, train_y))
print('Score de validação:',rf_clf.score(valid_X, valid_y))


plt.title('Com novas features a relação de importância ou correlação muda:')
sns.barplot(rf_clf.feature_importances_, feature_names);


# Melhoramos um pouco o nosso score de validação. Depois de incluir as novas features no modelo, agora Sexo passa a ser a feature mais importante.

# ## Feature Engineering - Título

# Feature Engineering é uma técnica que envolve criar novas features - em geral a partir de outras. Vamos usar essa técnica para extrair o título a partir do nome.
# Aqui vamos usar uma Regular Expression e a função apply() do pandas para transformar a coluna Name e extrair a informação do Título

# In[ ]:


titanic_df.head()['Name']


# In[ ]:


import re
def extract_title(name):
    x = re.search(', (.+?)\.', name)
    if x:
        return x.group(1)
    else:
        return ''


# In[ ]:


titanic_df['Name'].apply(extract_title).unique()


# In[ ]:


titanic_df['Title'] = titanic_df['Name'].apply(extract_title)
test_df['Title'] = test_df['Name'].apply(extract_title)


# In[ ]:


#imprimindo o novo dataset
titanic_df.head()


# ## OneHotEncoding

# Agora vamos trabalhar com features que são Multicategóricas, e nesse caso devem ser adequamente transformadas em uma matriz dos valores possíveis.
# Para cada valor possível teremos uma coluna binária, que será marcada como 1 quando ela for a valor da amostra, e zero quando não for.
# Em suma se uma coluna possui 6 valores possíveis, ela será transformada em 6 colunas binárias, uma para cada valor possível.

# In[ ]:


train_X.shape
titanic_df['Embarked']= titanic_df['Embarked'].fillna('Z')


# In[ ]:


#from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'Embarked']
dv = DictVectorizer()
dv.fit(titanic_df[feature_names].append(test_df[feature_names]).to_dict(orient='records'))
dv.feature_names_


# **Para o treino e validação do algorítimo utilizado é necessário fazer um split do dataset das amostras:**

# In[ ]:


train_X, valid_X, test_y, valid_y = train_test_split(dv.transform(titanic_df[feature_names].to_dict(orient='records')),
                                                     titanic_df['Survived'],
                                                     test_size=0.2,
                                                     random_state=42)


# In[ ]:


from sklearn.model_selection import train_test_split

rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=5)
rf_clf.fit(train_X, train_y)

print('Novo Score de Treino:', rf_clf.score(train_X, train_y))
print('Novo Score de Validação:', rf_clf.score(valid_X, valid_y))

plt.title('Novo gráfico de correlação das features:')
sns.barplot(rf_clf.feature_importances_, dv.feature_names_);


# ## Submissão do Arquivo

# In[ ]:


test_df['Fare'] = test_df['Fare'].fillna(0)
test_df['Embarked']= test_df['Embarked'].fillna('Z')


# Infelizmente no dataset de teste, um dos passageiros está com Fare vazio. :-(
# 
# Para conseguirmos evoluir, vamos setar o Fare vazio para 0.0

# Lembra que o skleanr trabalha com matrizes numpy, certo?

# In[ ]:


test_X = dv.transform(test_df[feature_names].to_dict(orient='records'))
print(test_X.shape)


# Legal. Temos 418 amostras. Vamos usar o nosso modelo pra prever a sobrevivência dessas 418 pessoas.

# In[ ]:


y_pred = rf_clf.predict(test_X)


# In[ ]:


y_pred.shape


# Ótimo! Já temos aquilo que precisávamos. Próximo passo agora é empacotar num arquivo CSV e submeter no Kaggle.

# In[ ]:


submission_df = pd.DataFrame()


# In[ ]:


submission_df['PassengerId'] = test_df['PassengerId']
submission_df['Survived'] = y_pred
submission_df


# In[ ]:


submission_df.to_csv('submit_final.csv', index=False)


# Por favor, anote aqui para referência: quanto foi o seu score de treinamento do modelo? E no dataset de Validação? Quanto foi o seu score na submissão do Kaggle?

# In[ ]:




