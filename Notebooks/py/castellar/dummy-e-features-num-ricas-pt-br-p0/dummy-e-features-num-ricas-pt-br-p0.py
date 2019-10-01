#!/usr/bin/env python
# coding: utf-8

# # Titanic - DummyClassifier e Features Numéricas
# Este notebook cria um modelo de benchmark baseado no dataset do Titanic e usando um DummyClassifier. Para esse caso específico, estamos usando apenas features numéricas e o nosso modelo faz previsões baseado na classe predominante.

# Vamos começar importando as bibliotecas básicas que vamos usar.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Próximo passo: carregando os dados a partir dos CSVs disponibilizados no Kaggle. Estamos usando a biblioteca pandas para esse propósito.

# In[ ]:


# Vamos iniciar o notebook importanto o Dataset
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# Podemos observar as primeiras linhas dele.
titanic_df.head()


# Vamos usar a biblioteca scikit-learn para treinar um modelo. Como a maior parte das bibliotecas para Machine Learning, o tratamento é de dados numéricos. Dessa forma, vamos isolar as colunas numéricas. As colunas não numéricas serão tratadas no futuro.
# 
# A Feature `Age` também é numérica, mas algumas linhas não estão preenchidas. Vamos tratá-la posteriormente junto com as colunas não numéricas.
# 

# In[ ]:


numeric_features = ['Pclass', 'SibSp', 'Parch', 'Fare']


# É possível isolar apenas um subcojunto das colunas do DataFrame pandas, passando uma lista como index da subscription.

# In[ ]:


titanic_df[numeric_features].head()


# Apesar de usarmos pandas para ler e manipular os dados, a biblioteca scikit-learn trabalha os dados como se fossem matrizes e arrays numpy. Vamos portanto converter os dados que nos interessam em matrizes e arrays numpy.
# 
# Como convenção, normalmente representa-se as features do modelo com o nome `X` e a variável alvo, que estamos tentando prever, de `y`.

# In[ ]:


train_X = titanic_df[numeric_features].as_matrix()
print(train_X.shape)
train_y = titanic_df['Survived'].as_matrix()
print(train_y.shape)


# Ok. A matriz `train_X` tem 891 linhas e 4 colunas. O array `train_y` tem 891 valores, representando cada uma das 891 amostras. Vamos visualizá-los, apenas para fins didáticos.

# In[ ]:


train_X


# In[ ]:


train_y


# Agora que temos os dados, vamos criar o modelo mais bobo possível. Nosso objetivo é nos habituarmos com o workflow e a API do SKLearn, nem tanto com a qualidade do modelo.
# 
# Vamos construir um modelo que responde sempre a mesma coisa: a classe predominante.
# 
# O que é a classe predominante? Bem... nosso problema só tem 2 classes nesse caso: ou o passageiro é um sobervivente (Survived=1) ou não é um sobrevivente (Survived=0). Nesse dataset, infelizmente a maior parte das pessoas não sobreviveu.

# In[ ]:


import seaborn as sns
sns.countplot(titanic_df['Survived']);


# Mas nós não vamos dizer isso pra classificador! Nós vamos passar os dados pra ele e ele vai aprender qual é a classe predominante. O método que faz o aprendizado (ou treinamento) no sklearn é o fit.
# 
# Vamos começar então criando a estrutura do modelo.

# In[ ]:


from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy='most_frequent')


# Criamos a estrutura básica do modelo. Hora de treiná-lo. Vamos alimentá-lo com as features (características de cada passageiro) e o rótulo (ou alvo) - que determina se aquele dado passageiro sobreviveu ou não. Vamos usar o método fit.

# In[ ]:


dummy_clf.fit(train_X, train_y)


# Com nosso modelo treinado, o próximo passo é avaliar a qualidade dele. Esse modelo de Classifier usa como métrica de score a acurácia, ou seja: qual a taxa de acerto. Vamos ver a acurácia nesse dataset de treinamento.

# In[ ]:


dummy_clf.score(train_X, train_y)


# Bem, esse vai ser nosso benchmark inicial. O modelo mais simples possível (que prevê sempre a classe predominante) tem ~61% de acurácia. Já é melhor que cara-ou-coroa. Nossa meta é avançar.

# Agora que temos o nosso modelo treinado, como fazemos pra fazer uma previsão? Basicamente nós temos que informar as características do passageiro e ele vai nos dizer se o passageiro sobreviveu ou não. Isso é feito usando o método predict. 
# 
# Pra fazer um breve teste, vamos prever com esse modelo os últimos 5 passageiros desse dataset.

# In[ ]:


train_X[-5:]


# In[ ]:


train_y[-5:]


# In[ ]:


dummy_clf.predict(train_X[-5:])


# Ok, já sabemos como usar o modelo para prever se um determinado passageiro sobreviveria, com base em algumas de suas características (features). O próximo passo é trabalhar com o dataset de teste que o Kaggle nos disponibiliza. Vamos extrair as features (características) desse dataset, passar pelo modelo e os resultados serão submetidos no Kaggle.

# In[ ]:


test_df.head()


# O dataset de test é muito parecido com o dataset de treinamento (ainda bem!). A diferença aqui é que falta a coluna Survived, que é justamente a que precisamos prever. Vamos começar a preparar os dados.

# Infelizmente no dataset de teste, um dos passageiros está com Fare vazio. :-(
# 
# Para conseguirmos evoluir, vamos setar o Fare vazio para 0.0

# In[ ]:


test_df['Fare'] = test_df['Fare'].fillna(0)


# Lembra que o sklean trabalha com matrizes numpy, certo?

# In[ ]:


test_X = test_df[numeric_features].as_matrix()
print(test_X.shape)


# In[ ]:


test_X


# Legal. Temos 418 amostras. Vamos usar o nosso modelo pra prever a sobrevivência dessas 418 pessoas.

# In[ ]:


y_pred = dummy_clf.predict(test_X)


# In[ ]:


y_pred


# Ótimo! Já temos aquilo que precisávamos. Não é muito impressionante, mas nosso objetivo é testar o fluxo. Próximo passo agora é empacotar num arquivo CSV e submeter no Kaggle.

# In[ ]:


sample_submission_df = pd.DataFrame()


# In[ ]:


sample_submission_df['PassengerId'] = test_df['PassengerId']
sample_submission_df['Survived'] = y_pred
sample_submission_df


# In[ ]:


sample_submission_df.to_csv('basic_dummy_classifier.csv', index=False)


# Por favor, anote aqui para referência: quanto foi o seu score de treinamento do modelo? Quanto foi o seu score na submissão do Kaggle?
# 0.62679

# 
