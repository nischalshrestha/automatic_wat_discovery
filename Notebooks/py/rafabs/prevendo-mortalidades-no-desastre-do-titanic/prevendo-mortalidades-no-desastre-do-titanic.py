#!/usr/bin/env python
# coding: utf-8

# # Prevendo Mortalidades no Desastre do Titanic
# 
# Esse é o resultado dos meus esforços na primeira competição sugerida no [Kaggle](https://www.kaggle.com), sobre prever quem sobreviveu no desastre do Titanic a partir de informações sobre a tripulação.
# 
# Me baseei intensamente nos resultados obtidos dos outros participantes. Principalmente [nesse Kernel do Sinakhorami](https://www.kaggle.com/sinakhorami/titanic-best-working-classifier). Vou refazer os passos dele, traduzindo e adicionando observações minhas.
# 
# ## Carregando bibliotecas

# In[ ]:


# carregando bibliotecas

import pandas as pd
import numpy as np

import re
import math

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# carregando bibliotecas

import pandas as pd
import numpy as np

import re
import math

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')


# ## Carregando dados e entendendo o problema
# 
# O náufrago do RMS Titanic é um dos mais infames da história. No dia 15 de Abril de 1912, durante sua viagem inaugural, o Titanic afundou após colidir com um iceberg, matando 1502, dos 2224 passageiros e tripulação. Essa incrível tragédia chocou a comunidade internacional e nos levou a melhores regulações de segurança para os navios.
# 
# Uma das razões de que o naufrago levou a tantas mortes é que não haviam botes salva vidas o bastante para os passageiros e tripulação. Além disso, existe uma dose de sorte envolvida na sobrevivência. Alguns grupos de pessoas são mais prováveis a sobreviver que outros, tais como mulheres, crianças e a classe alta.
# 
# Nesse desafio, devo fazer uma análise completa de quais tipos de pessoas são mais prováveis de sobreviver. Em particular, devo usar Machine Learning (clusterring) para prever quais passageiros sobreviveram à tragédia (prever a variável Survived dos dados de teste), a partir dos dados abaixo.

# In[ ]:


# carregando dados
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")

# guardando alguns dados que podem ser importantes no futuro
train_y = train["Survived"]
PassengerId = test['PassengerId']


# #### Dados de treino

# In[ ]:


train.head()


# #### Dados de teste

# In[ ]:


test.head()


# ## Analise exploratória e limpeza dos dados
# 
# Antes de mais nada, é sempre bom fazer a análise exploratória para termos uma ideia da distribuição e características das variáveis que estão sendo trabalhadas.
# 
# ### Todas as variáveis

# In[ ]:


train.info()


# Todas as variáveis já estão no tipo esperado. As colunas numéricas estão no formato "int" ou "float" e as categóricas no formato "object". Temos dados faltando na coluna Age, Cabin e Embarked. Na coluna Cabin mais da metade dos dados estão faltando, o que torna essa variável muito pouco útil, então vou simplesmente removê-la. Vou remover também as colunas PassangerId e Ticket, que não trazem nenhuma informação relevante sobre a sobrevivência.

# In[ ]:


train = train.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
test = test.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)

train.head(3)


# ## Name
# 
# Por senso comum, não tem porque acreditarmos que o nome da pessoa influencia a probabilidade de sobrevivência dela, mas dessa variável nós podemos extrair o título de cada pessoa.

# In[ ]:


full_data = [train, test]

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

pd.crosstab(train['Title'], train['Sex']).plot(kind='bar', 
                                                    title ="Sexo por Title",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# Tudo parece certo até aqui. Os títulos masculinos de fato são de pessoas do sexo masculino, e os femininos, de pessoas do sexo feminino. Agora que temos os títulos, vamos categoriza-los e checar se eles têm algum impacto na taxa de sobrevivência.

# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train = train.drop(["Name"], axis = 1)
test = test.drop(["Name"], axis = 1)

pd.crosstab(train['Title'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Title",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# In[ ]:


train.head(3) # acompanhando o progresso


# ## Pclass
# 
# Essa variável representa o status socioeconômico da pessoa, onde:
# 
# 1 = Alta
# 
# 2 = Média
# 
# 3 = Baixa
# 
# Ela não tem nenhum valor faltando. Vou observar quais são os valores únicos e checar o impacto da variável nos dados de treino, para me certificar de que está tudo certo.

# In[ ]:


print("Valores únicos na coluna Pclass: " + str(np.sort(train.Pclass.unique())))
print(" ")
print( "Quantidade de pessoas na classe 1: " + str(train["Pclass"][train["Pclass"] == 1].count()))
print( "Quantidade de pessoas na classe 2: " + str(train["Pclass"][train["Pclass"] == 2].count()))
print( "Quantidade de pessoas na classe 3: " + str(train["Pclass"][train["Pclass"] == 3].count()))
print(" ")
print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# In[ ]:


train.head(3)


# ## Sex
# 
# Vou exibir os Valores Únicos da variável e checar o impacto nos dados de treino.

# In[ ]:


print("Valores únicos na coluna Sex: " + str(train.Sex.unique()))
print(" ")
print( "Quantidade total de mulheres: " + str(train["Sex"][train["Sex"] == "female"].count()))
print( "Quantidade total de homens: " + str(train["Sex"][train["Sex"] == "male"].count()))

pd.crosstab(train['Sex'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Sex",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# Sex é uma variável categórica. Aqui eu tenho duas opções que dão o mesmo resultado. A primeira é substituir todos os valores que estão faltando pelo valor mais provável ("male") e depois mapear os dados, substituindo 1 para homens e 0 para mulheres. A segunda, que tem o mesmo efeito, é usar a função "get_dummies" e excluir a coluna "Sex_male". Eu pretendo lidar com todas as variáveis categóricas ao mesmo tempo, mais para frente, usando "get_dummies".

# In[ ]:


train.head(3)


# ## SibSp e Parch
# 
# SibSp representa a quantidade de irmãos/cônjuges da pessoa que também estão no barco, tal que:
# Irmãos = irmão, irmã, meio-irmão, meia-irmã
# Cônjuge = Marido, Esposa (amantes e noivas foram ignorados)
# 
# Parch representa a quantidade de pais/filhos da pessoa que também estão no barco, tal que:
# Pais = mãe, pai
# Filho = filho, filha, enteada, enteado
# Algumas crianças viajam somente com as babás, então Parch = 0 para eles.
# 
# Vou mostrar algumas informações sobre essas variáveis também.

# In[ ]:


print("Valores únicos na coluna SibSp: " + str(np.sort(train.SibSp.unique())))
pd.crosstab(train['SibSp'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por SibSp",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# In[ ]:


print("Valores únicos na coluna Parch: " + str(np.sort(train.Parch.unique())))
pd.crosstab(train['Parch'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Parch",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# Com a quantidade de irmãos/cônjuges e o número de filhos/pais, nós podemos criar uma nova variável chamada Family Size, que representa a quantidade de membros da família que a pessoa tem viajando com ela no navio e medir o impacto dessa nova variável na taxa de sobrevivência.

# In[ ]:


full_data = [train, test]

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
pd.crosstab(train['FamilySize'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por FamiliSize",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# Ela parece ter um bom efeito na nossa predição. Vamos também checar quem está sozinho no barco.

# In[ ]:


for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
pd.crosstab(train['IsAlone'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por IsAlone",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# Nesse último gráfico, para a variável IsAlone, 1 representa as pessoas que estão sozinhas, e 0 as que não estão sozinhas. Já para a variável Survived, 1 representa que a pessoa sobreviveu e 0 que ela morreu.
# 
# Nós podemos concluir que estar acompanhado aumenta significativamente a probabilidade de sobrevivência.

# In[ ]:


train.head(3)


# ## Embarked
# 
# A variável Embarked representa o porto no qual a pessoa embarcou no navio, onde:
# C = Cherbourg
# 
# Q = Queenstown
# 
# S = Southampton

# In[ ]:


print("Valores únicos na coluna Embarked: " + str(train.Embarked.unique()))
print(" ")
print( "Quantidade de pessoa que embarcaram em S: " + str(train["Embarked"][train["Embarked"] == "S"].count()))
print( "Quantidade de pessoa que embarcaram em C: " + str(train["Embarked"][train["Embarked"] == "C"].count()))
print( "Quantidade de pessoa que embarcaram em Q: " + str(train["Embarked"][train["Embarked"] == "Q"].count()))
pd.crosstab(train['Embarked'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por Embarked",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# ## Lidando com variáveis categóricas
# 
# Vou aplicar o método "One Hot Encoder", usando a função get_dummies para transformar as variáveis Sex, Embarked e Title em variáveis numéricas. Vou concatenar os conjuntos de dados, usar o get_dummies, depois separar. Essa não é a melhor prática, pois torna mais complicado de transformar qualquer outro "test set" que poderíamos usar no futuro, mas como meu objetivo é só trabalhar com os dados de treino e teste atuais, não devo ter maiores problemas com isso.
# 
# O método "One Hot Encoder" cria novas colunas para cara valor único de uma variável, e mapeia de tal forma que 1 representa que a pessoa tem aquele valor, e 0 representa que não tem. Por exemplo, se a pessoa embarcou no porto Q, ela terá os valores: Embarked_S = 0; Embarked_C = 0; Embarked_Q = 1.
# 
# A variável Embarked tem alguns valores faltando. Remover a coluna "Embarked_S" será equivalente a substituir os valores faltando pelo valor mais provável ('S'). A lógica é que as informações da coluna S já estão representadas nas outras duas colunas. Se a pessoa não embarcou no posto C, nem no Q, então ela deve ter embarcado no S. O mesmo vale para a variável Sex, onde vou remover a coluna "Sex_Male".

# In[ ]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

data = pd.concat(( train, test ))

data = pd.get_dummies(data)
data = data.drop(['Embarked_S', 'Sex_male'], axis = 1)

train = data.iloc[:891,:]
test = data.iloc[891:,:]

train.head(3)


# ## Fare
# 
# Fare representa a tarifa de passageiros. Ela é a variável mais problemática. Quero conseguir ver a relação entre Fare e Survived, para isso vou plotar, no mesmo gráfico, a variável Fare das pessoas que sobreviveram, com a cor azul, e as que morreram, com a cor vermelha.

# In[ ]:


fig, ax_lst = plt.subplots(1, 2, figsize=(15, 10))
ax_lst[0].plot(train["Fare"][train["Survived"] == 1], "b.", train["Fare"][train["Survived"] == 0], "r.")
train.boxplot(column="Fare",by="Survived", ax = ax_lst[1])
fig.suptitle('Survived por Fare')


# Quanto maior for o valor da variável Fare, maiores são as chances da pessoa sobreviver, mas os dados estão com muito ruído. Para lidar com o ruído, vou separar os dados em classes. Vou também substituir os valores que estão faltando pela mediana, e não pela média, já que temos outliers.

# In[ ]:


full_data = [train, test]

k = round(3.322 * math.log10(train.shape[0]) + 1) # fórmula de Sturges

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'], Fare_labels = pd.qcut(train['Fare'], k, retbins = True)
train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()


# Agora está muito mais fácil de ver a diferença na taxa de sobrevivência, tanto para nós, quanto para o algoritmo de classificação.

# In[ ]:


for dataset in full_data:
    for i in range(len(Fare_labels)-1):
        dataset.loc[(dataset['Fare'] > Fare_labels[i]) & (dataset['Fare'] <= Fare_labels[i+1]), 'Fare'] = i

train = train.drop(['CategoricalFare'], axis = 1)

train.head(3)


# ##  Age
# 
# Age representa a idade das pessoas em anos, mas se a idade é menor que 1, ela é representada por um valor fracionado.

# In[ ]:


fig, ax_lst = plt.subplots(1, 2, figsize=(15, 10))
ax_lst[0].plot(train["Age"][train["Survived"] == 1], "b.", train["Age"][train["Survived"] == 0], "r.")
train.boxplot(column="Age",by="Survived", ax = ax_lst[1])
fig.suptitle('Survived por Age')


# Os dados estão cheios de ruído, o que torna muito difícil ver qualquer relação entre a idade e a taxa de sobrevivência.
# 
# Vou criar uma nova variável, chamada IsChild, que dirá se a pessoa é criança (menor de 14 anos) ou não.

# In[ ]:


full_data = [train, test]

for dataset in full_data:
    dataset['IsChild'] = dataset['Age'].apply(lambda x: 1 if x < 14 else 0)

pd.crosstab(train['IsChild'], train['Survived']).plot(kind='bar', 
                                                    title ="Sobrevivencia por IsChild",
                                                    figsize=(10, 5),
                                                    legend=True,
                                                    fontsize=14)


# Apesar de termos um número muito maior de adultos do que de crianças, conseguimos perceber que a probabilidade de sobrevivência das crianças é consideravelmente maior.
# 
# Aqui não parece que temos os problemas de outliers que tivemos na variável Fare, por isso vou substituir os dados que estão faltando pela média e depois separar os dados em classes.

# In[ ]:


full_data = [train, test]

for dataset in full_data:
    dataset['Age'] = dataset['Age'].fillna(train['Age'].mean())
train['CategoricalAge'], Age_labels = pd.qcut(train['Age'], k, retbins = True, duplicates='drop')
train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()


# In[ ]:


for dataset in full_data:
    for i in range(len(Age_labels)-1):
        dataset.loc[(dataset['Age'] > Age_labels[i]) & (dataset['Age'] <= Age_labels[i+1]), 'Age'] = i

train = train.drop(['CategoricalAge'], axis = 1)

train.head(3)


# # Últimos detalhes
# 
# Quero saber o quão relacionada cara variável é com as outras. Para isso, vou usar o a biblioteca Seaborn, que me permite gerar heatmaps de correlações convenientemente.

# In[ ]:


colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
plt.show


# Uma informação que podemos tirar desse gráfico de correlações é que não temos muitas variáveis fortemente correlacionadas com as outras. Isso é bom para quando for usa-las nos modelos, porque significa que não temos dados redundantes/supérfluos no conjunto de treino. Para garantir que cada variável carrega uma informação única, vou remover colunas de forma que termine sem nenhuma que tenha mais do que 75% de correlação com as outras.

# In[ ]:


train = train.drop(["SibSp", "Parch", "Title_Mr"], axis = 1)
test = test.drop(["SibSp", "Parch", "Title_Mr"], axis = 1)

train.head()


# ## Normalizando os dados

# In[ ]:


train_X = train.drop(["Survived"], axis = 1)
test_X = test.drop(["Survived"], axis = 1)

SScaler = StandardScaler()
SScaler  = SScaler.fit(train_X)

train_X = SScaler.transform(train_X)
test_X = SScaler.transform(test_X)


# ### Rodando o algoritmo de classificação
# 
# Agora basta alimentar qualquer algoritmo de classificação, como o Support Vector Classiffier, para obtermos o conjunto de respostas. Com o que foi feito até aqui eu já consegui 0.79425, ou 79,425% de precisão no Kaggle, o que me fez ficar em 2471 lugar entre 11280 times. Estou bastante satisfeito com os resultados, mas anda existem muitas coisas que podem ser feitas para melhorar a precisão, como testar diversos algoritmos, com diferentes parâmetros, para descobrir qual funciona melhor.
# 
# A tabela abaixo mostra o ID de cada pessoa e a sobrevivência prevista para cada uma, onde 1 significa que a pessoa sobrevive e 0 significa que ela morre.

# In[ ]:


modelo = SVC()
modelo.fit(train_X, train_y) # Já tenho "train_y" desde o começo do código

predict = modelo.predict(test_X)
predict_dict = pd.DataFrame({'PassengerId': PassengerId, "Survived": predict})
predict_dict


# Se quiser rodar somente o código no python, você pode encontra-lo [aqui](https://github.com/1rafa/Prevendo-Mortalidades-no-Desastre-do-Titanic).
# 
# Obrigado por ter lido até aqui.

# 

# 

# 

# 
