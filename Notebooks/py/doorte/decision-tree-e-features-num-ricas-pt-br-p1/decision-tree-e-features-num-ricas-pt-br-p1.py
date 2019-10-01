#!/usr/bin/env python
# coding: utf-8

# # Titanic - DecisionTree e Features Numéricas
# This is an educational kernel for Brazilian Portuguese Speakers using Decision Trees and Numeric Features.
# 
# Este notebook cria um modelo baseado no dataset do Titanic e usando uma DecisionTree. Para esse caso específico, estamos usando apenas features numéricas.

# Vamos começar importando as bibliotecas básicas que vamos usar.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:





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


# Ótimo! Vamos agora trabalhar com o nosso modelo. Nesse caso específico, vamos usar uma DecisionTree.
# 
# O parâmetro random_state é para garantir que sempre que executarmos esse código tenhamos os mesmos resultados. O parâmetro max_depth é um hiperparâmetro ajustável, com o qual vamos brincar.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='entropy')


# Criamos a estrutura básica do modelo. Hora de treiná-lo.

# In[ ]:


dt_clf.fit(train_X, train_y)


# Com nosso modelo treinado, vamos avaliar a qualidade dele. Esse modelo de DecisionTree usa como métrica de score a acurácia, ou seja: qual a taxa de acerto.

# In[ ]:


dt_clf.score(train_X, train_y)


# Quer ver um efeito colateral bem legal de se usar uma Decision Tree? Ele consegue nos mostrar quais as features mais importantes para o modelo. Isso pode nos ajudar a entender o que está acontecendo melhor e explicar para o nosso cliente, chefe, etc. Não é possível fazer isso com todo tipo de modelo (redes neurais, por exemplo).

# In[ ]:


dt_clf.feature_importances_


# In[ ]:


# Se voce estiver usando linux e tiver curiosidade em ver a arvore...
#from sklearn.tree import export_graphviz
#export_graphviz(dt_clf, feature_names=numeric_features, out_file='/tmp/x.dot', filled=True, class_names=True, impurity=False, proportion=True)
#!dot -Tpng /tmp/x.dot -o /tmp/x.png


# Nesse caso específico, qual é a feature mais importante do nosso modelo? Isso faz sentido pra você?

# ## Exercício
# Você consegue melhorar a acurácia desse modelo? Faça testes usando diferentes valores para max_depth. Trace um gráfico com max_depth variando entre 1 e 20. No eixo X do gráfico, qual o parâmetro max_depth usado. No eixo y a acurácia. Abaixo, um starter code para você brincar. ;)

# In[ ]:


# cria um array cuja posição 1 é 1, posição 2 é 2, ...
max_depth_arr = np.arange(1, 6)
# criar um array com 1000 posições zeradas
accuracy_arr = np.zeros(5)

for i, max_depth in enumerate(max_depth_arr):
    ## calcula accuracy usando o max_depth em questao
   dt_clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth, criterion='entropy')
   dt_clf.fit(train_X, train_y)
   accuracy_arr[i] = dt_clf.score(train_X, train_y)


plt.plot(max_depth_arr, accuracy_arr);    


# In[ ]:





# Pergunta: qual o melhor max_depth que você encontrou? Vamos usá-lo em seguida.

# In[ ]:


optimal_max_depth = 5 # coloque aqui o max_depth que voce encontrou


# In[ ]:


dt_clf = DecisionTreeClassifier(random_state=42, max_depth=optimal_max_depth)
dt_clf.fit(train_X, train_y)


# Agora que temos o nosso modelo já com o hiperparâmetro tunado, vamos nos preparar para submeter o arquivo. Antes de mais nada: como fazemos pra fazer o nosso modelo "prever" um resultado? Vamos usar o próprio dataset de treino para ver.

# In[ ]:


train_X[0:5]


# In[ ]:


train_y[0:5]


# In[ ]:


dt_clf.predict(train_X[0:5])


# Ok, já sabemos como usar o modelo para prever se um determinado passageiro sobreviveria, com base em algumas de suas características (features). O próximo passo é trabalhar com o dataset de teste que o Kaggle nos disponibiliza.

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


y_pred = dt_clf.predict(test_X)


# In[ ]:


y_pred


# Ótimo! Já temos aquilo que precisávamos. Próximo passo agora é empacotar num arquivo CSV e submeter no Kaggle.

# In[ ]:


sample_submission_df = pd.DataFrame()


# In[ ]:


sample_submission_df['PassengerId'] = test_df['PassengerId']
sample_submission_df['Survived'] = y_pred
sample_submission_df


# In[ ]:


sample_submission_df.to_csv('basic_decision_tree.csv', index=False)


# Por favor, anote aqui para referência: quanto foi o seu score de treinamento do modelo? Quanto foi o seu score na submissão do Kaggle?

# In[ ]:




