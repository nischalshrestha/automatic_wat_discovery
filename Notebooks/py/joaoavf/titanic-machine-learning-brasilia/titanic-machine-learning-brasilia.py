#!/usr/bin/env python
# coding: utf-8

# ## Fazendo as importações necessárias

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# ## Preparando o MatPlotLib

# In[ ]:


# Alguns computadores precisam disso para mostrar os gráficos
get_ipython().magic(u'matplotlib inline')


# ## Lendo os arquivos train.csv e test.csv:

# O train.csv contém as colunas que iremos usar para fazer as previsões, e também a resposta de quem sobreviveu ou não.

# In[ ]:


df = pd.read_csv('../input/train.csv')


# O test.csv contém apenas as colunas que iremos usar para fazer as previsões. **Não** contém a resposta de quem sobreviveu ou não, quem tem essa resposta é o Kaggle.
# 
# Por esse motivo chamaremos esse arquivo de `kaggle_sub`, já que o conjunto de dados que será submetido ao Kaggle.
# 
# A única função desses dados é servir de insumo para fazermos as previsões que serão enviadas ao Kaggle (que é a última parte deste caderno).

# In[ ]:


kaggle_sub = pd.read_csv('../input/test.csv')  # Optei por usar o nome de variável kaggle_sub, para diferenciar de outra variável chamada test que criei mais a frente


# Verificando o tamanho do `df`:

# In[ ]:


df.shape


# Temos 891 linhas e 12 colunas:

# ## Juntando os nossos 2 arquivos em um único DataFrame

# Este passo é importante para podermos fazer a codifição de texto para números e mantermos os mesmos códigos identificadores para os dois conjuntos de dados (`df` e `kaggle_sub`):
# 
# 

# In[ ]:


df = df.append(kaggle_sub, sort=False)  # O sort=False, serve para retirar um Warning do pandas


# Podemos verificar o tamanho do novo objeto `df`:

# In[ ]:


df.shape


# Temos 1309 linhas e as mesmas 12 colunas que tínhamos anteriormente. O nosso agrupamento de `DataFrames` deu certo!

# ## Criando Novas Colunas (Feature Engineering) 
# ## *(Vá direto para Pré-Processamento de Dados caso esse seja o primeiro contato com o caderno)*

# Abaixo seguem algumas possibilidades de colunas novas a serem criadas a partir do conjunto original de dados.
# 
# Optei por deixar o código desta parte todo comentado, porque o ideal seja que isso seja executado em um segundo passo da análise, logo após ter um modelo funcionando para poder comparar se as novas colunas de fato melhoram o desempenho.
# 
# Se for a primeira vez rodando o código, pode pular até chegar em **Pré-processamento de Dados**.

# Tirei algumas ideias de features deste site:
#     
# https://triangleinequality.wordpress.com/2013/09/08/basic-feature-engineering-with-the-titanic-data/ 

# ### Sobrenomes:

# Uma hipótese é que pessoas com certos sobrenomes tenham sido favorecidas ou desfavorecidas no processo de seleção de sobreviventes, já que isso pode conter uma série de informações sociais e econômicas sobre essas pessoas.

# In[ ]:


#df['Surname'] = df['Name'].str.split(',').str[0]


# ### Títulos:

# Alguns nomes possuem algum tipo de título junto, como `Rev`, `Mr`, `Master`. Existe uma hipótese de que pessoas com títulos de nobreza tenham uma maior chance de sobreviver. Para isso podemos usar o comando abaixo para manipular o texto e separar apenas o título:

# In[ ]:


#df['Title'] = df['Name'].str.split(',').str[1].str.split().str[0]  


# ### Cabines:

# Vamos imprimir as linhas 27 a 31 do DataFrame em relação a coluna `Cabin`:

# In[ ]:


df['Cabin'][27:32]


# Como podemos ver acima, temos 2 tipos diferentes de linhas, a linha 27 que contém 3 cabines, e a linha 31 que contém 1 cabine. A variável a seguir serve para criar uma nova variável com a contagem do total de cabines a fim de avaliar se isso tem alguma relevância:

# In[ ]:


#df['Cabin Len'] = df.Cabin.str.split().str.len()


# Outra possível análise é saber qual a letra da cabine. Cada letra correspondia a uma região no navio, o que pode ter sido de grande importância para as probabilidades de sobrevivência.

# In[ ]:


#df['Cabin Letter'] = df['Cabin'].str[0]


# ### Tamanho da Família

# Segundo o site do Kaggle, a coluna `SibSp` corresponde a quantidade de irmãos ou esposos a bordo, e a coluna `Parch` corresponde a quantidade de pais e filhos a bordo. Tudo isso em relação a pessoa a qual corresponde a linha.
# 
# Uma hipótese é o tamanho da família influencia nas chances de sobrevivência porque pode fazer com que uma pessoa vá atrás de outras pessoas, ou que um grupo de pessoas consiga exercer pressão para priorizar o salvamento de uma criança ou mulher da família.
# 
# Mesmo sem criar uma nova coluna, essa informação já está disponível por meio das colunas `SibSp` e `Parch`,  mas é possível que o nosso modelo encontra formas mais fáceis de generalizar com uma coluna contendo a soma de `SibSp` e `Parch`, que é o teste que iremos fazer.

# In[ ]:


#df['Family_Size'] = df['SibSp'] + df['Parch']


# ### Ticket por Membro da Família

# Se observarmos abaixo as pessoas do mesmo sobrenome, no caso `Andersson`, veremos que tem várias pessoas pessoas com o mesmo ticket, e o mesmo preço pago pelo ticket (que é a coluna `Fare`).

# In[ ]:


df[df['Name'].str.contains('Andersson,')]


# A coluna `Fare` não nos diz em média quanto cada pessoa pagou pelo ticket, apenas quanto custou aquele ticket no total, independentemente se 1 ou 10 pessoas usaram ele.
# 
# E como uma pessoa que paga 30 por um ticket provavelmente tem um poder aquisitivo maior do que uma família de 6 pessoas que paga 30 por um ticket (equivale a um custo de 5 por pessoa), isso pode acabar sendo relevante para o nosso modelo.
# 
# É possível fazermos uma aproximação calculadando o `Fare` dividido pelo `Family_Size` (coluna construída artificialmente por nós)

# In[ ]:


#df['Fare Per Person'] = df['Fare'] / (df['Family_Size'] + 1)


# ### Calculando Quantas Pessoas Usam um Mesmo Ticket

# Uma forma mais robusta de calcularmos o preço do Ticket individual, é calculando quantas pessoas usaram um mesmo Ticket, e depois fazer o preço do Ticket dividido pelo número de pessoas.
# 
# Para isso vamos criar uma nova coluna que contabilize quantas pessoas usaram um mesmo ticket:

# In[ ]:


#df['Number of Ticket Uses'] = df.groupby('Ticket', as_index=False)['Ticket'].transform(lambda s: s.count())


# ### Calculando Custo Médio por Ticket

# Basta dividimos o preço do Ticket, que é a coluna `fare`, pela coluna `Number of Ticket Uses`:

# In[ ]:


#df['Average Fare per Person'] = df['Fare'] / df['Number of Ticket Uses'] 


# ## Pré-processamento de Dados

# Primeiro, vamos transformas as colunas que são texto em número:

# In[ ]:


for col in df.columns:  # Loop usado para avaliar todas as colunas
    if df[col].dtype == 'object':  # No pandas, 'object' é usado para tudo que não é datetime ou número, e em 99.9% dos usos de caso mais comuns vai ser relativo a texto.
        df[col] = df[col].astype('category')  # Transforma texto em categoria
        df[col] = df[col].cat.codes  # Salva apenas os números que correspondiam aos códigos de cada categoria


# Vamos substituir os valores em branco por `-1` (com RandomForest funciona bem substituir os valores em branco por um valor menor que a mínima ou maior que a máxima dos seus dados):

# In[ ]:


df.fillna(-1, inplace=True)


# ### Restaurando os DataFrames Originais

# Vamos voltar os dados para os duas variáveis originais, `df` e `kaggle_sub`:

# In[ ]:


kaggle_sub = df.iloc[891:].copy()
df = df.iloc[:891].copy()


# ### Criando os Conjuntos de Treino e Teste

# Vamos importar o `train_test_split` do `sklearn` para podermos dividir o nosso `df` em treino e teste:

# In[ ]:


from sklearn.model_selection import train_test_split


# Vamos dividir o nosso `df` em treino e teste, usando 20% (0.2) dos dados para teste, e setando um estado aleatório fixo para podermos replicar os resultados em outras iterações:

# In[ ]:


train, test = train_test_split(df, test_size=0.2, random_state=42)


# ### Criando o RandomForestClassifier

# Vamos criar o nosso objeto de `RandomForestClassifier`: (note que nesse caso se trata de uma Classificação para dizer se a pessoa sobreviveu ou não, por isso o uso do Classifier em vez do Regressor que foi utilizado pelo Jeremy no fastai)

# In[ ]:


rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features=.5, random_state=42)


# ### Selecionando as Colunas que Serão Analisadas

# Vamos selecionar as colunas as quais vamos alimentar o nosso modelo:

# In[ ]:


remove = ['Survived', 'PassengerId', 'Name']   # Optei por retiras as colunas PassengerId e Name porque não há motivo para elas terem fator preditivo
feats = [col for col in df.columns if col not in remove]  # Crio a lista de colunas que serão usadas


# ### Treinando o Modelo

# Treino o `RandomForestClassifier`:

# In[ ]:


rf.fit(train[feats], train['Survived'])


# ### Fazendo as previsões do Modelo:

# Faço as previsões do conjunto `traina`:

# In[ ]:


preds_train = rf.predict(train[feats])


# Faço as previsões do conjunto `test`:

# In[ ]:


preds = rf.predict(test[feats])


# ### Avaliando o Desempenho do Modelo

# Importo o `accuracy_score` para avaliarmos o desempenho do nosso modelo:

# In[ ]:


from sklearn.metrics import accuracy_score


# Avalio o desempenho do treino:

# In[ ]:


accuracy_score(train['Survived'], preds_train)


# Avalio o desempenho do teste:

# In[ ]:


accuracy_score(test['Survived'], preds)


# # Criando o Arquivo de Submissão ao Kaggle

# Crio uma nova RandomForest para alimentar com os dados de treino e teste, para fazer previsões no `kaggle_sub` e em seguida enviar ao Kaggle.

# In[ ]:


rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features=.5, random_state=42)


# Treino o modelo com todos os dados do `df`:

# In[ ]:


rf.fit(df[feats],df['Survived'])


# Crio as previsões no conjunto de dados `kaggle_sub`:

# In[ ]:


preds_kaggle = rf.predict(kaggle_sub[feats])


# Crio um arquivo chamado `Submission.csv` com as submissões para serem enviadas ao Kaggle.

# In[ ]:


submission = pd.DataFrame({ 'PassengerId': kaggle_sub['PassengerId'],
                            'Survived': preds_kaggle }, dtype=int)
submission.to_csv("submission.csv",index=False)


# In[ ]:




