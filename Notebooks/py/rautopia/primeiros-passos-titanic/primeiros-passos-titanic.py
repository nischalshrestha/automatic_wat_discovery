#!/usr/bin/env python
# coding: utf-8

# > 
# Este tutorial orienta você ao enviar um arquivo ".csv" de previsões para Kaggle pela primeira vez.<br><br>
# 
# ### Pontuação e desafios:<br>
# 
# Se você simplesmente executar o código abaixo, sua pontuação será bastante baixa. Eu deixei intencionalmente muito espaço para melhorias em relação ao modelo usado (atualmente um simples classificador de árvore de decisão). <br> <br> A idéia deste tutorial é começar e tomar as decisões de como melhorar sua pontuação. Na parte inferior do tutorial, há desafios que, se você segui-los, melhorarão significativamente sua pontuação.
# 
# 
# 
# ### Etapas para concluir este tutorial no seu próprio computador:
# O kernel abaixo pode ser executado no navegador. Mas se você quiser executar o código localmente em seu próprio computador, siga as etapas abaixo.
# 1. Crie uma conta Kaggle (https://www.kaggle.com/).
# 2. Download do conjunto de dados do Titanic (https://www.kaggle.com/c/titanic/data).<br>
#     a. Download 'train.csv' and 'test.csv'.<br>
#     b. Coloque os dois arquivos em uma pasta chamada 'input'.<br>
#     c. Coloque essa pasta no mesmo diretório do seu notebook.
# 3. Instale [Jupyter Notebooks](https://jupyter.org/) (Siga minha [installation tutorial](http://joshlawman.com/getting-set-up-in-jupyter-notebooks-using-anaconda-to-install-the-jupyter-pandas-sklearn-etc/)se você está confuso)
# 4. Baixe este kernel como um [notebook](https://github.com/jlawman/Meetup/blob/master/11.7%20Meetup%20-%20Decision%20Trees/Submit%20your%20first%20Kaggle%20prediction%20-%20Titanic%20Dataset.ipynb) with empty cells from my GitHub. If you are new to GitHub go [the repository folder](https://github.com/jlawman/Meetup), clique "Clone or Download", 
# em seguida, descompacte o arquivo e retire o bloco de anotações desejado.
# 5. Corra cada célula do caderno (except the optional visualization cells).
# 6. Envie o arquivo CSV contendo as previsões.
# 7. Tente melhorar a previsão usando as solicitações de desafio adequadas ao seu nível.

# ## 1. Process the data
# 
# ### Load data

# In[ ]:


#carregando arquivo
import pandas as pd
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#prestem atenção no código abaixo, tem duas formas de carregar os csvs, esolham a que preferir e comentem a outra pra desativar
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
#train = pd.read_csv(train_url)

#test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
#test = pd.read_csv(test_url)


#Solte os recursos que não usaremos
train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#Veja as 3 primeiras linhas dos nossos dados de treinamento
train.head(3)


# 
# Nossos dados possuem as seguintes colunas:
# - PassengerId - O id de cada passageiro
# - Survived - Se o passageiro sobreviveu ou não (1 - sim, 0 - não)
# - Pclass - A classe de passageiros: (1 ª classe - 1, 2ª classe - 2, 3ª classe - 3)
# - Sex - Sexo de cada passageiro
# - Age - A idade de cada passageiro

# ### Prepare os dados para serem lidos pelo nosso algoritmo

# In[ ]:


#Converter ['male','female'] para [1,0] para que nossa árvore de decisão possa ser construída
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
#Preencha os valores de idade ausentes com 0 (presumindo que sejam bebês se não tiverem uma idade listada)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Observe as primeiras 3 linhas (temos mais de 800 linhas no total) dos nossos dados de treinamento. 
#Este é o "input" que nosso classificador usará como "input"
train[features].head(3)


# Vamos examinar as três primeiras variáveis-alvo correspondentes. Esta é a medida de se o passageiro sobreviveu ou não (i.e. o primeiro passageiro(22 de idade do sexo masculino) não sobreviveu, 
# mas o segundo passageiro (38 anos de idade do sexo feminino sobreviveram).
# <br><br>
# Nosso classificador usará isso para saber qual deve ser a saída para cada uma das instâncias de treinamento.

# In[ ]:


train.info()
print('_'*40)
test.info()


# In[ ]:


#Exibe as primeiras 3 variáveis de destino
train[target].head(3).values


# [](http://)# 2. Crie e ajuste a árvore de decisão
# 
# 
# Esta árvore definitivamente vai sobrecarregar nossos dados. Quando chegar ao estágio de desafio, você pode retornar aqui e ajustar os hiperparâmetros nesta célula. Por exemplo, você pode reduzir a profundidade máxima da árvore para 3 definindo max_depth=3 com o seguinte comando:
# >clf = DecisionTreeClassifier(max_depth=3)
# 
# Para alterar vários hiperparâmetros, separe os parâmetros com uma vírgula.Por exemplo, para alterar a taxa de aprendizado e amostras "samples" mínimas por folha "leaf" e a profundidade máxima, preencha os parênteses com o seguinte:
# >clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf=2)
# 
# Os outros parâmetros estão listados abaixo.
# Você também pode acessar a lista de parâmetros lendo o [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) 
# para classificadores de árvore de decisão. Outra maneira de acessar os parâmetros é colocar o cursor entre os parênteses e pressionar a tecla shift.
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

#Criar um objeto classificador com hiperparâmetros padrão
#clf = DecisionTreeClassifier()  
clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf=2)
#Ajuste nosso classificador usando os recursos de treinamento e os valores de meta de treinamento
clf.fit(train[features],train[target]) 


# ****### Visualize a árvore padrão (opcional)
# Este não é um passo necessário, mas mostra quão complexa é a árvore quando você não a restringe. Para completar esta seção de visualização, você deve estar passando pelo código em seu computador.

# In[ ]:


#Create decision tree ".dot" file

#Remove each '#' below to uncomment the two lines and export the file.
#from sklearn.tree import export_graphviz
#export_graphviz(clf,out_file='titanic_tree.dot',feature_names=features,rounded=True,filled=True,class_names=['Survived','Did not Survive'])


# ****NNote, se você quiser gerar uma nova árvore png, você precisa abrir o terminal (ou prompt de comando) depois de executar a célula acima. Navegue até o diretório em que você possui este bloco de notas e digite o seguinte comando
# >dot -Tpng titanic_tree.dot -o titanic_tree.png<br><br>

# In[ ]:


#Display decision tree

#Blue on a node or leaf means the tree thinks the person did not survive
#Orange on a node or leaf means that tree thinks that the person did survive

#In Chrome, to zoom in press control +. To zoom out, press control -. If you are on a Mac, use Command.

#Remove each '#' below to run the two lines below.
#from IPython.core.display import Image, display
#display(Image('titanic_tree.png', width=1900, unconfined=True))


# > # 3. Fazer previsões
# 

# In[ ]:


#Faça previsões usando os recursos do conjunto de dados de teste
predictions = clf.predict(test[features])

#Exibir nossas previsões - elas são 0 ou 1 para cada instância de treinamento 
#dependendo se nosso algoritmo acredita que a pessoa sobreviveu ou não.
predictions


# # 4. Crie o csv para fazer o upload para o Kaggle

# In[ ]:


#Crie um DataFrame com os IDs dos passageiros e nossa previsão sobre se eles sobreviveram ou não
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize as primeiras 5 linhas
submission.head()


# In[ ]:


#Converter DataFrame em um arquivo csv que pode ser carregado
#Isso é salvo no mesmo diretório do seu notebook
filename = 'Titanic Predictions 2.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# **# 5. Enviar arquivo para o Kaggle
# 
# Vá até [submission section](https://www.kaggle.com/c/titanic/submit) do concurso Titanic. Arraste seu arquivo do diretório que contém seu código e faça sua submissão.<br><br> 
# Parabéns - você está no placar!****

# # Desafios
# 
# A árvore de decisão padrão dá uma pontuação de 0,70813 colocando você na posição 8,070 de 8.767. Você pode melhorar isso?
# 
# ### Level 1: First time on Kaggle
# 
# Nível 1a: Você pode tentar dar à árvore uma profundidade máxima para melhorar sua pontuação?
# 
# Level 1b:  Você pode importar diferentes modelos de árvore, como o Random Forest Classifier para ver como isso afeta sua pontuação? Use a seguinte linha de código para criá-lo. Compare este modelo com uma árvore de decisão com depth 3.
# > from sklearn.ensemble import RandomForestClassifier<br>
# > clf = RandomForestClassifier() ****
# 
# 
# ### Level 2: Enviado para Kaggle antes
# Level 2a: Você pode incluir outros recursos que foram descartados para melhorar sua pontuação? Não se esqueça de lidar com quaisquer dados perdidos.
# <br><br>
# Level 2b: Você consegue visualizar seus dados usando matplotlib ou seaborn para obter outras ideias de como melhorar suas previsões?
# 
# ### Level 3: Alguma familiaridade com o scikit-learn
# Level 3a: Você pode usar GridSearchCV de sklearn.model_selection no Random Forest Classifier para ajustar os hyperparameters e melhorar sua pontuação?
# <br><br>
# Level 3b: Você pode treinar uma lista de modelos e, em seguida, avaliar cada um usando a função sklearn.metrics train_test_split para ver qual lhe dá a melhor pontuação?
# <br><br>
# Level 3c: Você pode pegar a lista do desafio 3b e depois ter os melhores modelos da lista votando sobre como cada predição deve ser feita?
