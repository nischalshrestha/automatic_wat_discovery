#!/usr/bin/env python
# coding: utf-8

# # Titanic: Solução simplificada para Iniciantes utilizando Redes Neurais Densas
#  **[Renan Gomes Barreto](https://www.kaggle.com/renangomes)**  -  *Maio, 2018*
# 
# ![Titanic](https://media.giphy.com/media/Uj3SeuVfg2oCs/giphy.gif)
# 
# ## Introdução
# 
# Este notebook contém uma introdução breve sobre como criar uma rede neural de múltiplas camadas e resolver o problema do Titanic usando um modelo simples no Keras. Modelos Densos, também chamados de Multi-layer perceptrons (MLP), podem ser usados como base e, a partir deles, serem construídos modelos mais complexos.
# 
# Neste notebook criaremos um classificador binário utilizando os dados de passageiros do Titanic. Com esses dados, desejamos predizer se o passageiro irá sobreviver ou não ao naufrágio.
# ****
# Para resolver problemas com Redes Neurais, assim como qualquer problema de aprendizado de máquina, antes de elaborar uma arquitetura é necessário primeiramente entender o problema e, principalmente, compreender os dados. Esse processo é extenso e compreende muitas vezes a maior parte do trabalho. Dessa forma, esse notebook tem como objetivo introduzir alguns pontos importantes da pirâmide para obtenção de melhores resultados de uma rede neural. O notebook é organizado da seguinte forma:
# 
# * [Introdução](#Introdução)
# * [Definição do problema](#Definição-do-problema)
# * [Carregando o dataset](#Carregando-o-dataset)
# * [Pré-processamento](#Pré-processamento)
# * [Implementando a Rede Neural](#Implementando-a-Rede-Neural)
# * [Resultados](#Resultados)
# * [Conclusão](#Conclusão)

# ## Definição do problema
# 
# Nesse problema, será utilizado como base de dados informações dos passageiros do Titanic para identificar quais passageiros sobreviveram. No Titanic, uma das razões que causou o naufrágio  foi que não havia botes salva-vidas suficientes para os passageiros e a tripulação. Dentre os passageiros, alguns grupos de pessoas tinham maior probabilidade de sobreviver do que outros, como mulheres, crianças e a classe alta. Dessa forma, o problema consiste em utilizar rede neural para identificar quais pessoas poderiam sobreviver.

# ## Carregando o Dataset

# ### Lendo os arquivos do Dataset
# 
# Para iniciar, deve-se analisar os atributos de entrada do dataset, seus tipos e o atributo alvo (label/rótulo). Isso pode ser feito através do Pandas, biblioteca de Python específica para análise e pré-processamento de dados.

# In[1]:


import numpy as np
np.random.seed(10)

import pandas as pd 

train = pd.DataFrame(pd.read_csv("../input/train.csv", index_col=[0], header=0))
test  = pd.DataFrame(pd.read_csv("../input/test.csv", index_col=[0], header=0))
display(train.head())


# ## Pré-processamento
# 
# As colunas Name, Ticket e Cabin parecem ser características exclusivas do passageiro e por isso iremos descarta-las. Em uma análise mais profunda, certamente utilizaríamos essas colunas para melhorar os dados ou até deduzir dados faltantes.

# In[2]:


train.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
test.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
display(train.head())


# ### Tratando os dados faltantes
# 
# Dados faltantes são um problema grave em aprendizagem de máquina. De alguma forma deveremos trata-los. A forma mais fácil de trata-los é simplesmente excluindo todas as linhas do dataset que possuem esses dados ou substituindo-os por um valor fixo. No nosso caso, para as colunas numéricas Age e Fare, substituímos os dados faltantes pela média. Já as colunas SibSp e Parch tiveram seus dados substituídos por -1.

# In[16]:


train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Fare'].fillna(train['Fare'].mean(), inplace=True)
train['SibSp'].fillna(-1, inplace=True)
train['Parch'].fillna(-1, inplace=True)

test['Age'].fillna(train['Age'].mean(), inplace=True)
test['Fare'].fillna(train['Fare'].mean(), inplace=True)
test['SibSp'].fillna(-1, inplace=True)
test['Parch'].fillna(-1, inplace=True)


# ### Codificando as colunas categóricas
# 
# As colunas Pclass, Sex e Embarked parecem ser categóricas. Essas colunas devem ser mapeadas para números e, de preferência, cada possível valor deve se tornar uma nova coluna binária. Isso pode ser feito facilmente utilizando a função get_dummies do pandas.

# In[4]:


train = pd.get_dummies(train, dummy_na=True, columns=['Pclass', 'Sex', 'Embarked']).astype(float)
test = pd.get_dummies(test, dummy_na=True, columns=['Pclass', 'Sex', 'Embarked']).astype(float)

display(train.head())
display(test.head())


# ## Implementando a Rede Neural

# ### Separando os atributos da saída
# Os atributos de saída serão separados. Além disso, separamos o dataset de treinamento original em dois.

# In[5]:


X_train = train.drop(columns=["Survived"])[:-120]
y_train = train["Survived"][:-120]

X_val = train.drop(columns=["Survived"])[-120:]
y_val = train["Survived"][-120:]

X_test = test

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_val: ",   X_val.shape)
print("y_val: ",   y_val.shape)
print("X_test: ",   X_test.shape)


# ### Definição do Modelo
# 
# Criaremos um modelo simples usando Keras. Sinta-se livre para alterar a quantidade de neurônios, camadas, funções de ativação, etc.

# In[6]:


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.summary()


# ### Treinamento
# 
# Treinaremos a rede por 1750 épocas com batch size de 32. Caso você queira ver estatísticas durante o treinamento, ative o parâmetro verbose.

# In[7]:


import time

epochs = 1750
start_time = time.time()

history = model.fit(X_train.as_matrix(), y_train.as_matrix(), epochs=epochs, batch_size=32, 
                    validation_data=(X_val.as_matrix(), y_val.as_matrix()), verbose=0, shuffle=True)

print("Tempo gasto: %d segundos" % (time.time() - start_time), "\r\nÉpocas: %d" % (epochs))


# ### Gráficos da etapa de treinamento

# In[8]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'], color="r")
plt.plot(history.history['val_acc'], color="g")
plt.title('Curva de Treinamento')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='lower right')
plt.show()

plt.plot(history.history['loss'], color="r")
plt.plot(history.history['val_loss'], color="g")
plt.title('Curva de Treinamento')
plt.ylabel('Erro')
plt.xlabel('Época')
plt.legend(['Treinamento', 'Validação'], loc='upper left')
plt.show()


# ## Resultados

# ### Matriz de Confusão - Treinamento e Validação
# 
# A fim de entendermos a o resultado do treinamento, utilizaremos as funções accuracy_score e confusion_matrix da biblioteca sklearn.
# Lembre-se que as variáveis y_train e X_train são dataframe do Pandas, então, geralmente, vamos ter que utilizar a função as_matrix antes de usa-las.

# In[19]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np

print("Acurácia no Dataset de Treinamento:", accuracy_score(y_train.as_matrix(), np.round(model.predict(X_train.as_matrix()))), "\r\n")

confusionMatrixDF = pd.DataFrame( confusion_matrix(y_train.as_matrix(), np.round(model.predict(X_train.as_matrix()))),
                                 index=('Sobrevivente', 'Vítima'), columns=('Sobrevivente', 'Vítima'))

heatmap = sns.heatmap(confusionMatrixDF, annot=True, fmt="d", cmap="Blues",  vmin=0)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print("Acurácia no Dataset de Validação:", accuracy_score(y_val.as_matrix(), np.round(model.predict(X_val.as_matrix()))), "\r\n")

confusionMatrixDF = pd.DataFrame( confusion_matrix(y_val.as_matrix(), np.round(model.predict(X_val.as_matrix()))),
                                 index=('Sobrevivente', 'Vítima'), columns=('Sobrevivente', 'Vítima'))

heatmap = sns.heatmap(confusionMatrixDF, annot=True, fmt="d", cmap="Blues",  vmin=0)
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# ### Enviando os Resultados para o Kaggle

# In[23]:


y_test_pred = model.predict(X_test.as_matrix())

X_test_submission = X_test.copy()
X_test_submission['Survived'] = np.round(y_test_pred).astype(int)
X_test_submission['Survived'].to_csv('submission.csv', header=True)


# ## Conclusão
# 
# Neste notebook mostramos uma solução simples para o problema do Titanic.
# Foi implementado uma rede neural com duas camadas que obteve uma acurácia satisfatória no dataset de validação.
