#!/usr/bin/env python
# coding: utf-8

# # Titanic - RandomForest e Features Engineering
# Este notebook cria um modelo baseado no dataset do Titanic e usando RandomForests. Esse notebook trata de dados faltantes, features categóricas e um pouco de feature engineering.

# Vamos começar importando as bibliotecas básicas que vamos usar.

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Próximo passo: carregando os dados a partir dos CSVs disponibilizados no Kaggle. Estamos usando a biblioteca pandas para esse propósito.

# In[6]:


# Vamos iniciar o notebook importanto o Dataset
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# Podemos observar as primeiras linhas dele.
titanic_df.head()


# Vamos começar com o básico de tratamento desse dataset. Importante: tudo que fizermos vamos fazer no dataset de treinamento e também de teste.

# ## Tratando a Idade - Imputation

# Nos casos anteriores, avisamos que idade tinha campos vazios.

# In[7]:


titanic_df[pd.isnull(titanic_df['Age'])]


# Teremos que preencher isso de algum jeito. Uma abordagem comum nesses casos é usar uma média ou mediana. Vamos usar aqui a mediana do dataset - mas poderíamos agrupar por sexo, por exemplo. Fica a seu critério fazer isso de forma mais fancy. ;)

# In[8]:


age_median = titanic_df['Age'].median()
print(age_median)


# In[9]:


titanic_df['Age'] = titanic_df['Age'].fillna(age_median)
test_df['Age'] = test_df['Age'].fillna(age_median)


# Ótimo! Um problema a menos. Essa técnica que usamos é chamada de "imputation".

# ## Tratando Gênero - LabelEncoding

# Próximo passo: vamos tratar das features categóricas. Queremos transformar features que são de múltiplas categorias em dados numéricos que podemos trabalhar. O caso mais óbivo é sexo.

# In[10]:


import seaborn as sns
sns.countplot(titanic_df['Sex']);


# Vamos usar aqui um transformer do sklearn chamado LabelEncoder. Ele transforma a primeira categoria no número 0, a segunda no número 1 e assim por diante.

# In[11]:


from sklearn.preprocessing import LabelEncoder
sex_encoder = LabelEncoder()

sex_encoder.fit(list(titanic_df['Sex'].values) + list(test_df['Sex'].values))


# In[12]:


sex_encoder.classes_


# In[13]:


titanic_df['Sex'] = sex_encoder.transform(titanic_df['Sex'].values)
test_df['Sex'] = sex_encoder.transform(test_df['Sex'].values)


# In[14]:


sns.countplot(titanic_df['Sex']);


# Ok, a feature Sex já está devidamente encodada. Vamos dar mais uma espiada nos dados?

# In[15]:


titanic_df.head()


# Já temos mais colunas numéricas. Vamos estudar o impacto de adicionar essas colunas no nosso modelo. Vamos usar o nosso modelo anterior.

# In[16]:


feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare']


# In[17]:


from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(titanic_df[feature_names].as_matrix(), 
                                                      titanic_df['Survived'].as_matrix(),
                                                      test_size=0.2,
                                                      random_state=42)
                                                      
                                                      
print(train_X.shape)
print(valid_X.shape)                                           
print(train_y.shape)
print(valid_y.shape)


# In[18]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=7)
rf_clf.fit(train_X, train_y)
print(rf_clf.score(train_X, train_y))
print(rf_clf.score(valid_X, valid_y))


# In[19]:


import seaborn as sns
sns.barplot(rf_clf.feature_importances_, feature_names);


# Vamos incluir algumas features adicionais

# In[20]:


feature_names = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age', 'Sex']

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(titanic_df[feature_names].as_matrix(), 
                                                      titanic_df['Survived'].as_matrix(),
                                                      test_size=0.2,
                                                      random_state=42)
                                                      
                                                      
print(train_X.shape)
print(valid_X.shape)                                           
print(train_y.shape)
print(valid_y.shape)

rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)
rf_clf.fit(train_X, train_y)
print(rf_clf.score(train_X, train_y))
print(rf_clf.score(valid_X, valid_y))

sns.barplot(rf_clf.feature_importances_, feature_names);


# Melhoramos um pouco o nosso score de validação. Depois de incluir as novas features no modelo, agora Sexo passa a ser a feature mais importante.

# ## Feature Engineering - Título

# Feature Engineering é uma técnica que envolve criar novas features - em geral a partir de outras. Vamos usar essa técnica para extrair o título a partir do nome.

# In[21]:


titanic_df.head()['Name']


# In[26]:


import re
def extract_title(name):
    x = re.search(', (.+?)\.', name)
    if x:
        return x.group(1)
    else:
        return ''


# In[27]:


titanic_df['Title'] = titanic_df['Name'].apply(extract_title)
test_df['Title'] = test_df['Name'].apply(extract_title)


# In[28]:


titanic_df.head()


# ## OneHotEncoding

# Agora vamos trabalhar com features que são MultiCategoricas. 

# In[ ]:





# In[29]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title']
dv = DictVectorizer()
dv.fit(titanic_df[feature_names].append(test_df[feature_names]).to_dict(orient='records'))
dv.feature_names_


# In[30]:


train_X, valid_X, test_y, valid_y = train_test_split(dv.transform(titanic_df[feature_names].to_dict(orient='records')),
                                                     titanic_df['Survived'],
                                                     test_size=0.2,
                                                     random_state=42)


# In[31]:


train_X.shape


# In[32]:


from sklearn.model_selection import train_test_split
rf_clf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)
rf_clf.fit(train_X, train_y)
print(rf_clf.score(train_X, train_y))
print(rf_clf.score(valid_X, valid_y))

sns.barplot(rf_clf.feature_importances_, dv.feature_names_);


# ## Exercício
# A coluna Embarked contém o porto de embarque do passageiro. Algumas linhas não estão preenchidas.

# In[33]:


titanic_df[pd.isnull(titanic_df['Embarked'])]


# - Implemente uma estratégia para fazer Imputation do porto de embarque desses passageiros. 
# - Em seguida, faça o OneHotEncoding para que eles entrem na lista de Features do Modelo. Essas novas features melhoram o modelo de alguma forma?
# - Crie uma nova feature, com o tamanho da familia. O tamanho da família é derivado de Parch e SibSp
# - Inclua essa nova feature no modelo. Ela melhora o modelo de alguma forma?

# ## Submissão do Arquivo

# In[34]:


train_X = dv.transform(titanic_df[feature_names].to_dict(orient='records'))
train_y = titanic_df['Survived']

rf_clf = RandomForestClassifier(random_state=42, max_depth=10, n_estimators=200)
rf_clf.fit(train_X, train_y)


# Infelizmente no dataset de teste, um dos passageiros está com Fare vazio. :-(
# 
# Para conseguirmos evoluir, vamos setar o Fare vazio para 0.0

# In[35]:


test_df['Fare'] = test_df['Fare'].fillna(0)


# Lembra que o sklean trabalha com matrizes numpy, certo?

# In[36]:


test_X = dv.transform(test_df[feature_names].to_dict(orient='records'))
print(test_X.shape)


# Legal. Temos 418 amostras. Vamos usar o nosso modelo pra prever a sobrevivência dessas 418 pessoas.

# In[37]:


y_pred = rf_clf.predict(test_X)


# In[38]:


y_pred


# Ótimo! Já temos aquilo que precisávamos. Próximo passo agora é empacotar num arquivo CSV e submeter no Kaggle.

# In[39]:


submission_df = pd.DataFrame()


# In[40]:


submission_df['PassengerId'] = test_df['PassengerId']
submission_df['Survived'] = y_pred
submission_df


# In[41]:


submission_df.to_csv('feature_engineering.csv', index=False)


# Por favor, anote aqui para referência: quanto foi o seu score de treinamento do modelo? E no dataset de Validação? Quanto foi o seu score na submissão do Kaggle?

# In[ ]:




