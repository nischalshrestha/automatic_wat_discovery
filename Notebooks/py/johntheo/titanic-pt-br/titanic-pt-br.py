#!/usr/bin/env python
# coding: utf-8

# **OBS.:**[EN] I create this notebook to brazilian folks that are interested in a introductory step by step guide

# # Classificação de Sobreviventes do Titanic
# 
# Este é um guia passo a passo para o problema de Classificação de Sobreviventes e enteder o funcionamento básico dos kernels do Kaggle
# 
# ## Configuração Inicial
# Nos próximos passos seguiremos com a configuração inicial do ambiente(no caso o kernel) para que possamos realizar a classificação
# 
# ### Importando as bibliotecas
# Para realizar a classificação, diversar bibliotecas irão auxiliar o nosso trabalho. Seja para ler arquivos, seja para realizar operações matemáticas e até mesmo para executar algoritmos de classificação. Sendo assim realizamos o **import** das seguintes biblitecas:
# * **numpy** para operações matemáticas
# * **pandas** para a leitura e manusei dos dados
# * **seaborn** e **matplotlib** para visualização
# * **sklearn** para utilização de modelos de classificação e demais ferramentas de aprendizado

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler        #Normalização dos dados
from sklearn.decomposition import PCA                   #Principal Component Analysis
from sklearn.model_selection import train_test_split    #Separação em treino e teste 
from sklearn.linear_model import LogisticRegression     #Modelo de classificação


# ### Lendo os valores

# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# 
# 

# ## Analisando e preparando os dados
# Nesta etapa iremos analisar cada um das variáveis de entrada e identificar o potencial e as falhas de cada uma. O Kaggle até fornece algo como "Column Metrics" porém não é muito confiável.
# 
# Temos os seguintes dados disponíveis:
# * **PassengerId**: Identificação do passageiro
# * **Survived**: Classificação se o passageiro sobreviveu ou não
# * **Pclass**: Classe que o passageiro estava ocupando
# * **Name**:  Nome do passageiro
# * **Sex**: Sexo do Passageiro
# * **Age**: Idade do passageiro
# * **SibSp**: Quantidade de irmãos e esposo(a) do passageiro
# * **Parch**:  Quantidade de pai/mãe e filhos do passageiro
# * **Ticket**: Número do Ticket
# * **Fare**: Tarifa do passageiro
# * **Cabin**: Cabine em que o passageiro ficou
# * **Embarked**: Portão em que o passageiro embarcou
# 
# ### 4 C's do tratamento dos dados
# Para tratar os dados devemos:
# * **Corrigir**: quando o dado vem com algum tipo de ruido que pode ser corrigido
# * **Completar**: quando podemos completar com valores conhecidos ou que podem ser obtidos através de média por exemplor
# * **Criar**: criar novas features que possam ter mais significancia
# * **Converter**: Quando um dado nao vem categorizado, por exemplo, e podemos converter em categorias os valores

# > AINDA EM CONSTRUCAO.....
