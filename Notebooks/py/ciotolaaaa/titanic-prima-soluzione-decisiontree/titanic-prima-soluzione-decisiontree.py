#!/usr/bin/env python
# coding: utf-8

# # TITANIC: MACHINE LEARNING FROM DISASTER
# ## A KAGGLE competition
# In questo progetto si andrà ad analizzare un dataset contenente tutte le informazioni relative ai passeggeri del Titanic, nave tristemente famosa in quanto è affondata provocando diverse centinaia di morti. Il dataset è un dataset pubblico scaricato da Kaggle (https://www.kaggle.com/c/titanic), che è stato reso disponibile in quanto oggetto di una delle competizioni iniziali per familiarizzare con il sistema.

# In[ ]:


#importiamo le librerie necessarie per l'analisi del dataset
import pandas as pd
# pandas è una libreria molto utilizzata in data analysis in quanto permette di gestire grosse quantità di dati in maniera veloce e inuitiva
#inoltre offre molte altre funzionalità integrate, come la creazione di grafici ecc-
import numpy as np
#numpy è una libreria utilizzata per un'insieme di operazioni numeriche
import matplotlib.pyplot as plt
#matplotlib ci permette di visualizzare l'andamento dei dati attraverso grafici e tabelle
import seaborn as sns
#seaborn, come matplotlib, permette di visualizzare grafici e creare istogrammi


# ### Il dataset è diviso in due parti: test.csv e train.csv. 
# Il primo viene usato come 'ground truth', ovvero come campo di allenamento dove allenare il classificatore, mentre nel secondo si andrà a testare il modello ricavato in precedenza per osservare l'accuratezza. \n Siccome questa è una competitions, il testing set è proposto senza *labels*, quindi non saarò in grado di definire l'accuratezza della classificazione sul testing set. Possiamo ovviare a questo dividendo il training set in due parti, come se fosse l'intero dataset, ma questo avverà in un secondo momento.

# In[ ]:


#carichiamo il dataset
train= pd.read_csv("../input/train.csv")
#train = pd.read_csv("train.csv")
train.head()


# #### Le feature all'interno del dataset si distribuiscono come :
# - PASSENGERID:
#     indica l'ID di ogni passeggero
# - PCLASS:
#     è il tipo di cabina affidato a ogni passeggero: è un indicatore dello stato socio economico dei passeggeri
#     si divide in 1 classe, seconda classe, terza classe. 
# - AGE:
#     è l'età di ogni passeggero, è di tipo float perchè quando è una stima viene posto come un numero decimale.
# - SIBSP:
#     definisce le relazioni familiari: indica quanti fratelli/sorelle o mogli/spose sul Titanic
# - PARCH:
#     definisce le relazioni familiari: indica quanti mamme/papà o figli sul Titanic
# - TICKET:
#     Indica il numero di ticket 
# - FARE:
#     Indica la tariffa
# - Cabin:
#     Indica il numero della cabina
# - Embarked:
#     Indica dove si è imbarcato il passeggero (C = Cherbourg, Q = Queenstown, S = Southampton)

# In[ ]:


#per rendere più facile la lettura, cambiamo i nomi delle colonne in lowercase
train.columns = [x.lower() for x in train.columns]
train.head(20)


# In[ ]:


#train.info ci permette di analizzare il dataset e capire quali dati possiede ogni colonna
train.info()


# ### Preprocessing dei dati: valori nulli
# Notiamo come nel dataset sono presenti dei valori nulli: prima di poter visualizzare i dati e prima di creare il nostro classificatore dobbiamo trasformarli in modo da poterli rappresentare.
# Le feature su cui dobbiamo lavorare sono Age, cabin e embraked: tutte le altre feature non hanno valori nulli

# In[ ]:


#L'idea principale che si utilizza quando si vanno a riempire valori vuoti, e che i 'segnaposto' che si vanno a inserire 
#non devono togliere o dare informazioni aggiuntive ai dati, ovvero 'snaturare' il dataset: infatti, se noi diamo valori
#che non c'entrano con quelli reali, è molto probabile che la classificazione avvenga non inerente con la realtà.
#Inserendo nei valori nulla la media delle età, ricaviamo un dato che non aggiunge nè toglie valore alla distribuzione dei dati,
#non alterandone l'andamento.
train['age'] = train['age'].fillna(train['age'].median())
train['fare'] = train['fare'].fillna(train['fare'].median())
#stessa cosa per embarked: siccome non sappiamo dove sono saliti i passeggeri, poniamo come segnapost la U di 'Unknown'
train['embarked'] = train['embarked'].fillna('S')
# per le cabine dobbiamo capire come si distribuiscono nel dataset:
train['cabin'] = train['cabin'].fillna('Unknown')


# Prima di poter fare una classificazione, tutti i dati all'interno del dataset dovranno essere di tipo numerico: le feature EMBARKED e  CABIN sono dati testuali: dovremo trasformali. 
# Utilizzeremo un metodo 'ad etichetta': daremo un numero ad ogni campo che sarà identificativo della sua classe (es. a tutti i passeggeri che saranno saliti a Southampton daremo come valore 1, che è identificativo della classe S nel dataset.)
# 

# In[ ]:


#per farlo utilizziamo una funzione, denominata def_embarked, e andremo a modificare il dataset attraverso l'operatore lambda
def def_embarked(point):
    if point == "S":
        return 1
    elif point == "Q":
        return 2
    elif point == "C":
        return 3
    else:
        return 0

train["embarked_"] = train.apply(lambda row:def_embarked(row["embarked"]),axis=1)

#per cabin il discorso è un po' diverso: ogni cabina è divisa in una lettera identificativa di una parte della nave più
#il numero di stanza reale. è necessario dividere in classi in base alla lettera che vi è davanti.
# identifichiamo la posizione di ogni cabina (se la hanno) all'interno della nave
def def_position(cabin):
    return cabin[:1]
train["Position"] = train.apply(lambda row:def_position(row["cabin"]), axis=1)
#value_counts() ci restituisce quanti valori ci sono all'interno di uan colonna:
train["Position"].value_counts()
#osserviamo 8 possibili classi, che andremo ad aggiungere al nostro dataset:
def def_cabin(pos):
    if pos == "C":
        return 1
    elif pos == "B":
        return 2
    elif pos == "D":
        return 3
    elif pos == "E":
        return 4
    elif pos == "F":
        return 5
    elif pos == "A":
        return 6
    elif pos == "G":
        return 7
    else: 
        return 0
train["cabin_"] = train.apply(lambda row:def_cabin(row["Position"]),axis=1)
#stessa cosa la effettuiamo con male o female
def def_sex(sex):
    if sex=="male":
        return 0
    else:
        return 1
train["sex_"] = train.apply(lambda row: def_sex(row["sex"]),axis = 1)


# ### Preprocessing dei dati: valori utili
# Il dataset contiene dati che potrebbero essere poco utili durante la classificazione: PASSENGERID e NAME probabilmente non sono utili per una classificazione: saranno feature che andremo a cancellare.
# Inoltre togliamo tutte quelle feature che sono state trasformate

# In[ ]:


train = train.drop(columns="passengerid")
train = train.drop(columns="name")
train = train.drop(columns = "embarked")
train = train.drop(columns = "cabin")
train = train.drop(columns= "Position")
train = train.drop(columns="sex")

train = train.drop(columns="ticket") #drop ma c'è da rivedere, perchè non capisco come funziona


# In[ ]:


train.info()


# ### Classificazione: Preparazione
# Prima di effettuare la classificazione, abbiamo necessità di creare i vari vettori contenenti i dati relativi al dataset:
#     - Target Names: 
#         - ovvero il nome delle etichette, nel nostro caso sopravvissuto o non sopravvissuto
#     - Feature Names:
#         - Ovvero i nomi delle feature: nel nostro caso pclass, age, sibsp, parch, fare, embarked_,cabin_sex_
#     - Data:
#         - Ovvero i dati relativi a ogni campo del dataset: quindi il valore che ha ogni feature in ogni riga del dataset
#     - Target:
#         - Ovvero l'etichetta di ogni riga del dataset, che può essere 0 o 1, ovvero sopravvissuto o non sopravvissuto.

# In[ ]:


x = []
for i in train["survived"]:
    if(i == 1):
        x.append("Survived")
    else:
        x.append("Not Survived")
titanic_target_names = np.asarray(x)
titanic_feature_names =  np.asarray(train.columns[1:])
train_ = train.drop(columns="survived")
titanic_data = np.asarray(train_.get_values())
titanic_target = np.asarray(train["survived"])
#con train_test_split dividiamo il nostro dataset in due parti: la prima che la utilizzeremo per il training, grande il 75% del totale,
#mentra la seconda la utilizzeremo per il testing, che è grande il 25% del totale. Ovviamente dividerà anche le etichette relative
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(titanic_data,titanic_target,random_state=1)


# ### Classificazione: Decision Tree Classifier
# Come primo classificatore utilizziamo DecisionTreeClassifier, che si basa sull'interrogazione del dataset in modo da poterlo dividere sempre più in maniera specifica in modo da poter classificare meglio i dati. 
# Per info: https://it.wikipedia.org/wiki/Albero_di_decisione

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train,y_train)
print("Accuracy on training set: {}".format(tree.score(X_train,y_train)))
print("Accuracy on the testing set: {}".format(tree.score(X_test,y_test)))


# ### Classificazione: Analisi dei risultati ottenuti

# Sul training set abbiamo un'accuratezza alta, circa del 91%, mentre sul testing set abbiamo un 73%. Il valore che ci interessa è ovviamente il secondo: vogliamo avere una accuratezza maggiore quando non si hanno il valore delle etichette. Andiamo ad osservare come si distribuisce l'albero e se è possibile aumentare il grado di accuratezza.

# ### Salvare l'albero e visualizzarlo

# In[ ]:


#graphviz è una libreria che serve per caricare grafici e salvarli. Nel nostro caso andiamo a salvare la raffigurazione
#dell'albero sopra creato e poi andremo a visualizzarlo nel kernel. C'è salvato anche nella cartella
import graphviz
from sklearn.tree import export_graphviz
export_graphviz(tree,out_file="tree.dot",class_names=["Survived","Not Survived"],feature_names=titanic_feature_names,impurity=False,filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# Osserviamo come ogni feature ha dato il suo contributo alla classificazione

# In[ ]:


import matplotlib.pyplot as plt
def plot_feature_importances(model):
    n_features = titanic_data.shape[1]
    plt.barh(range(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features), titanic_feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()
plot_feature_importances(tree)


# La feature principale su cui si è basata la classificazione è il genere, seguito poi da l'età . Questo conferma l'idea che prima si salvano le donne e i bambini. 

# Possiamo gestire la 'profondità' di un DecisionTree attraverso l'attributo max_depth. Con profondità si intende il numero di domande massimo che il modello fa la dataset: in questo modo si combatte l'overfitting, ovvero la generalizzazione ottenuta solo su dati del dataset.
# Ora andiamo a visualizzare l'andamento del grado di importanza di ogni feature in base alla profondità massima.

# In[ ]:


results = []
importances = []
max_ = [0,0]
for i in range(1,10):
    tree = DecisionTreeClassifier(max_depth=i,random_state = 1)
    tree.fit(X_train,y_train)
    if (tree.score(X_test,y_test) > max_[0]):
        max_ = [tree.score(X_test,y_test),i-1]
    results.append(tree.score(X_test,y_test))
    importances.append(tree.feature_importances_)


# In[ ]:


plt.plot([max_[1]],[max_[0]],marker='o',color="red")
plt.plot(results)
plt.title("Accuracy on max_depth")
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
plt.legend(["Max Accuracy: {0} with {1} depth".format(round(max_[0],2),max_[1])],loc=(1.04,0.5))
plt.show()


# Notiamo come l'accuratezza vari al variare del numero massimo di interrogazioni al dataset, con accuratezza massima a 2. Come varia l'importanza di ogni feature in base alla profondità? 

# In[ ]:


plt.plot(importances)
plt.legend(titanic_feature_names,loc=(1.04,0.05))
plt.title("Feature importances through max_depth")
plt.ylabel('Accuracy')
plt.xlabel('max_depth')
#plt.legend(["Max Accuracy: {0} with {1} depth".format(round(max_[0],2),max_[1])],loc=(1.04,0.5))
plt.show()


# Notiamo come maggiori sono le domande poste dal modello, più si stabilizzano le percentuali di importanza di ogni feature. Questo perchè il modello, all'aumentare della profondità, è più soggetto a overfitting.

# In[ ]:


plt.plot(importances)
plt.plot([max_[1]],[max_[0]],marker='o',color="red")
plt.plot(results, color="red")
plt.legend(titanic_feature_names,loc=(1.04,0.05))
plt.show()


# Dall'ultimo grafico notiamo come a max_depth 2 abbiamo la maggior accuratezza, con relativa importanza delle feature

# ### Random Forest Classifier

# Con random forest classifier si intede un classificatore che si basa non su un solo albero, ma su più di uno, in modo da cercare di combattere l'overfitting.
# Richiamando il classificatore, possiamo inserire il numero di alberi che vogliamo creare: andiamo ad analizzare un po' quello che succede.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=1,random_state=0)
forest.fit(X_train,y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train,y_train)))
print("Accuracy on testing set: {:.3f}".format(forest.score(X_test,y_test)))


# In[ ]:


plot_feature_importances(forest)


# Le feature sono un po' diverse da quelle del singolo decision tree (perchè?)

# In[ ]:


results_forest = []
importances_forest = []
max_forest = [0,0]
for i in range(1,10):
    forest = RandomForestClassifier(n_estimators=i,random_state=0)
    forest.fit(X_train,y_train)
    if (forest.score(X_test,y_test) > max_forest[0]):
        max_forest = [forest.score(X_test,y_test),i-1]
    results_forest.append(forest.score(X_test,y_test))
    importances_forest.append(forest.feature_importances_)


# In[ ]:


plt.plot([max_forest[1]],[max_forest[0]],marker='o',color="red")
plt.plot(results_forest)
plt.title("Accuracy on max_depth")
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.legend(["Max Accuracy: {0} with {1} n_estimators".format(round(max_forest[0],2),max_forest[1])],loc=(1.04,0.5))
plt.show()


# In[ ]:


plt.plot(importances_forest)
plt.plot([max_forest[1]],[max_forest[0]],marker='o',color="red")
plt.plot(results_forest, color="red")
plt.legend(titanic_feature_names,loc=(1.04,0.05))
plt.show()


# Come prima, andiamo a visualizzare accuratezza in base al numero di alberi e percentuale di importanza di ogni feature. L'accuratezza non varia di molto.

# ### Conclusioni

# Si sono utilizzati due tipi di classificatori: DecisionTree e RandomForest. Per entrambi si hanno avuto risultati differenti:
# 
# - Best DecisionTree (max_depth = x): 
#     - On training set: %
#     - On testing set: %
# - Best RandomForest (n_estimators = x): 
#     - On training set: %
#     - On testing set: %
# 
# Per entrambi si ha avuta una buona classificazione, con differenze sostanziali individuate sull'importanza data a ogni feature. (PERCHE'?)

# #### Last but not least: inviamo i nostri risultati a Kaggle
# Per inviare a Kaggle i risultati e osservare la bontà della nostra classificazione, bisogna riportare tutte le operazioni fatte sul training set al test: osserviamo come è strutturato il test set, e vediamo se dobbiamo apportare delle modifiche:

# In[ ]:


test = pd.read_csv("../input/test.csv")
test1 = pd.read_csv("../input/test.csv")
test.head()


# Il test set (come ci immaginavamo) è uguale al training set, senza però avere le etichette che dicono se la persona è sopravvissuta o meno: quello dobbiamo scoprirlo noi. Applichiamo ogni operazione fatta prima sul nuovo documento.

# In[ ]:


test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Cabin"] = test["Cabin"].fillna("C")
test["Embarked"] = test["Embarked"].fillna("U")
test["embarked_"] = test.apply(lambda row:def_embarked(row["Embarked"]),axis=1)
test["Position"] = test.apply(lambda row:def_position(row["Cabin"]), axis=1)
test["cabin_"] = test.apply(lambda row:def_cabin(row["Position"]),axis=1)
test["sex_"] = test.apply(lambda row: def_sex(row["Sex"]),axis = 1)
test = test.drop(columns="PassengerId")
test = test.drop(columns="Name")
test = test.drop(columns = "Embarked")
test = test.drop(columns = "Cabin")
test = test.drop(columns= "Position")
test = test.drop(columns="Sex")
test = test.drop(columns="Ticket")
test.head()


# Ora che abbiamo pronto il nostro dataset, possiamo entrare nella classificazione. Utilizzeremo quella che ci ha mostrato migliori risultati, ovvero decision tree con grado di profondità 2:

# In[ ]:


best_tree = DecisionTreeClassifier(max_depth=2,random_state = 1)
best_tree.fit(X_train,y_train)
pred = best_tree.predict(test)
d =  {'PassengerId' : test1["PassengerId"],'Survived' : pred}
prediction = pd.DataFrame(d,columns=["PassengerId","Survived"])
prediction.to_csv("Kaggle_first_try.csv",index=False)

