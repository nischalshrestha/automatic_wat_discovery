#!/usr/bin/env python
# coding: utf-8

# # Titanic - RandomForest e Features Numéricas
# Este notebook cria um modelo baseado no dataset do Titanic e usando RandomForests. Para esse caso específico, estamos usando apenas features numéricas.

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


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(titanic_df[numeric_features].as_matrix(), 
                                                      titanic_df['Survived'].as_matrix(),
                                                      test_size=0.2,
                                                      random_state=42)
                                                      
                                                      
print(train_X.shape)
print(valid_X.shape)                                           
print(train_y.shape)
print(valid_y.shape)


# Ok. A matriz `train_X` tem 712 linhas e 4 colunas. O array `train_y` tem 712 valores, representando cada uma das 712 amostras que foram separados para o treinamento. Vamos visualizá-los, apenas para fins didáticos.
# 
# 20% dos dados foram separados para validação - i.e. - tuning de parametros

# In[ ]:


train_X


# In[ ]:


train_y


# Ótimo! Vamos agora trabalhar com o nosso modelo. Nesse caso específico, vamos usar uma RandomForest.
# 
# O parâmetro random_state é para garantir que sempre que executarmos esse código tenhamos os mesmos resultados. O parâmetro n_estimator é um hiperparâmetro ajustável, com o qual vamos brincar. Também podemos brincar com max_depth, como fizemos com DecisionTree

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=5)


# Criamos a estrutura básica do modelo. Hora de treiná-lo.

# In[ ]:


rf_clf.fit(train_X, train_y)


# Com nosso modelo treinado, vamos avaliar a qualidade dele. Esse modelo de DecisionTree usa como métrica de score a acurácia, ou seja: qual a taxa de acerto.

# In[ ]:


print(rf_clf.score(train_X, train_y))
print(rf_clf.score(valid_X, valid_y))


# Do mesmo jeito que podemos fazer em uma DecisionTree, é possível extrair a importância das features pra poder explicar pro seu chefe...

# In[ ]:


rf_clf.feature_importances_


# Que tal um gráfico pra mostrar pro chefe?

# In[ ]:


import seaborn as sns
sns.barplot(rf_clf.feature_importances_, numeric_features);


# Na real, uma RandomForest é uma combinação de DecisionTreeClassifier. Tá aí a prova:

# In[ ]:


rf_clf.estimators_


# Qual é a vantagem então? Emsembling! Combinar vários modelos diferentes evita overfitting e suaviza os pontos fracos dos modelos individuais.

# ## Exercício
# Você consegue melhorar a acurácia desse modelo? Faça testes usando diferentes valores para max_depth e n_estimators. Se preferir, pode brincar com os outros parâmetros. Qual a melhor acurácia no dataset de validação que você conseguiu?

# Pergunta: qual o melhor max_depth e n_estimators que você encontrou? Vamos usá-lo em seguida.

# In[ ]:


optimal_max_depth = 5 # coloque aqui o max_depth que voce encontrou
optimal_n_estimators = 10 # coloque aqui o n_estimators que voce encontrou


# Vamos usar um truquezinho agora. Agora que já tunamos os parâmetros, vamos usar todos os dados pra treinar o modelo. Não faz sentido mais ter separação entre treino e validação.

# In[ ]:


rf_clf = RandomForestClassifier(random_state=42, max_depth=optimal_max_depth, n_estimators=optimal_n_estimators)
rf_clf.fit(titanic_df[numeric_features].as_matrix(), titanic_df['Survived'].as_matrix())


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


y_pred = rf_clf.predict(test_X)


# In[ ]:


y_pred


# Ótimo! Já temos aquilo que precisávamos. Próximo passo agora é empacotar num arquivo CSV e submeter no Kaggle.

# In[ ]:


submission_df = pd.DataFrame()


# In[ ]:


submission_df['PassengerId'] = test_df['PassengerId']
submission_df['Survived'] = y_pred
submission_df


# In[ ]:


submission_df.to_csv('basic_random_forest.csv', index=False)


# Por favor, anote aqui para referência: quanto foi o seu score de treinamento do modelo? E no dataset de Validação? Quanto foi o seu score na submissão do Kaggle?

# In[ ]:




