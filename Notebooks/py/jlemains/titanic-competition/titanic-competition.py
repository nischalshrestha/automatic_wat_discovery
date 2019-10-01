#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Définition-fonctions-utiles" data-toc-modified-id="Définition-fonctions-utiles-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Définition fonctions utiles</a></span></li><li><span><a href="#Intégration-et-compréhension-des-données" data-toc-modified-id="Intégration-et-compréhension-des-données-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Intégration et compréhension des données</a></span><ul class="toc-item"><li><span><a href="#Distribution-des-variables-entières" data-toc-modified-id="Distribution-des-variables-entières-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Distribution des variables entières</a></span><ul class="toc-item"><li><span><a href="#PassengerId" data-toc-modified-id="PassengerId-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>PassengerId</a></span></li><li><span><a href="#PClass" data-toc-modified-id="PClass-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>PClass</a></span></li><li><span><a href="#Age" data-toc-modified-id="Age-2.1.3"><span class="toc-item-num">2.1.3&nbsp;&nbsp;</span>Age</a></span></li><li><span><a href="#SibSp" data-toc-modified-id="SibSp-2.1.4"><span class="toc-item-num">2.1.4&nbsp;&nbsp;</span>SibSp</a></span></li><li><span><a href="#Parch" data-toc-modified-id="Parch-2.1.5"><span class="toc-item-num">2.1.5&nbsp;&nbsp;</span>Parch</a></span></li><li><span><a href="#Fare" data-toc-modified-id="Fare-2.1.6"><span class="toc-item-num">2.1.6&nbsp;&nbsp;</span>Fare</a></span></li></ul></li><li><span><a href="#Distribution-des-variables-object" data-toc-modified-id="Distribution-des-variables-object-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Distribution des variables object</a></span><ul class="toc-item"><li><span><a href="#Sex" data-toc-modified-id="Sex-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Sex</a></span></li><li><span><a href="#Cabin" data-toc-modified-id="Cabin-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>Cabin</a></span></li><li><span><a href="#Ticket" data-toc-modified-id="Ticket-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>Ticket</a></span></li><li><span><a href="#Embarked" data-toc-modified-id="Embarked-2.2.4"><span class="toc-item-num">2.2.4&nbsp;&nbsp;</span>Embarked</a></span></li></ul></li></ul></li><li><span><a href="#Gestion-des-valeurs-manquantes" data-toc-modified-id="Gestion-des-valeurs-manquantes-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Gestion des valeurs manquantes</a></span><ul class="toc-item"><li><span><a href="#Cabin" data-toc-modified-id="Cabin-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Cabin</a></span></li><li><span><a href="#Age" data-toc-modified-id="Age-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Age</a></span></li><li><span><a href="#Embarked" data-toc-modified-id="Embarked-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Embarked</a></span></li><li><span><a href="#Fare" data-toc-modified-id="Fare-3.4"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>Fare</a></span></li></ul></li><li><span><a href="#Feature-Engineering" data-toc-modified-id="Feature-Engineering-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Feature Engineering</a></span><ul class="toc-item"><li><span><a href="#Catégories-d'age" data-toc-modified-id="Catégories-d'age-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Catégories d'age</a></span></li><li><span><a href="#Catégories-de-Fare" data-toc-modified-id="Catégories-de-Fare-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Catégories de Fare</a></span></li><li><span><a href="#Taille-de-la-famille" data-toc-modified-id="Taille-de-la-famille-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Taille de la famille</a></span></li></ul></li><li><span><a href="#Conversion-ou-Suppression-de-colonnes" data-toc-modified-id="Conversion-ou-Suppression-de-colonnes-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conversion ou Suppression de colonnes</a></span><ul class="toc-item"><li><span><a href="#Mapping-de-Name" data-toc-modified-id="Mapping-de-Name-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Mapping de Name</a></span></li><li><span><a href="#Encoding" data-toc-modified-id="Encoding-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Encoding</a></span></li><li><span><a href="#Création-d'une-colonne-de-Cluster" data-toc-modified-id="Création-d'une-colonne-de-Cluster-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Création d'une colonne de Cluster</a></span></li></ul></li><li><span><a href="#Features-Importances" data-toc-modified-id="Features-Importances-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Features Importances</a></span></li><li><span><a href="#Machine-Learning" data-toc-modified-id="Machine-Learning-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Machine Learning</a></span><ul class="toc-item"><li><span><a href="#Essai-avec-Pipeline" data-toc-modified-id="Essai-avec-Pipeline-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Essai avec Pipeline</a></span><ul class="toc-item"><li><span><a href="#Random-Forest" data-toc-modified-id="Random-Forest-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>Random Forest</a></span></li><li><span><a href="#KNN" data-toc-modified-id="KNN-7.1.2"><span class="toc-item-num">7.1.2&nbsp;&nbsp;</span>KNN</a></span></li><li><span><a href="#Support-Vector-Machine" data-toc-modified-id="Support-Vector-Machine-7.1.3"><span class="toc-item-num">7.1.3&nbsp;&nbsp;</span>Support Vector Machine</a></span></li><li><span><a href="#Decision-Tree" data-toc-modified-id="Decision-Tree-7.1.4"><span class="toc-item-num">7.1.4&nbsp;&nbsp;</span>Decision Tree</a></span></li><li><span><a href="#Logistic-Regression" data-toc-modified-id="Logistic-Regression-7.1.5"><span class="toc-item-num">7.1.5&nbsp;&nbsp;</span>Logistic Regression</a></span></li><li><span><a href="#On-regarde-les-scores-de-tous-les-modèles" data-toc-modified-id="On-regarde-les-scores-de-tous-les-modèles-7.1.6"><span class="toc-item-num">7.1.6&nbsp;&nbsp;</span>On regarde les scores de tous les modèles</a></span></li></ul></li></ul></li></ul></div>

# In[77]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(12,6)})


# # Définition fonctions utiles

# In[162]:


from sklearn.neighbors import KNeighborsRegressor
from pandas.api.types import is_numeric_dtype
import copy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Affichage d'un tableau avec le pourcentage des valeurs manquantes et le type de chaque variables
def print_MV_percentage(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()*100/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total de VM', 'Pourcentage'])
    missing_data = missing_data[missing_data['Total de VM']!=0]

    long=[]
    for i in range (0, len(missing_data.index)):
        long.append((data[missing_data.index[i]]).dtype)
    missing_data['Type'] = long
    return missing_data

# Méthode qui remplie les valeurs manquantes grace à l'algorithme de KNN (uniquement pour les variables numériques)
def fill_MV_KNN(data):
    data_num = data.select_dtypes(exclude=['object'])
    for col in data.columns:
        neighR = KNeighborsRegressor(n_neighbors=2)
        if data[col].isnull().sum() != 0:
            #subset of columns with no missing values
            columns_no_nan = data_num.dropna(axis=1, how='any').columns
            # X to predict
            X_pred = data[data[col].isnull()][columns_no_nan]
            # X for training
            X = data[columns_no_nan]
            # y with no MV
            y_full = data[col].dropna()
            #index of no missing values
            index = y_full.index
            #fit known rows
            found=False
            if is_numeric_dtype(data[col]):
                neighR.fit(X.loc[index],y_full)
                pred = neighR.predict(X_pred)
                found=True
            if found != False: 
                #create data frame with the prediction
                df_pred = pd.DataFrame(data = pred,index = X_pred.index,columns = ['value'])
                #Fill:
                for i in df_pred.index:
                    data.at[i,col]=df_pred.at[i,'value']

# Encodage des données
def encoding(data):
    #colonnes catégorielles
    df = copy.deepcopy(data)
    for i in df.select_dtypes(include=['object']).columns:
        list_unique = set(df[i].unique())
        dict_pro = dict(zip(list_unique,np.arange(len(list_unique))))
        df[i] = df[i].map(dict_pro)
    return df

# Affichage de l'importance des variable grace à un arbre ExtraTrees
def feature_importance (data,target,text):
    from sklearn.ensemble import ExtraTreesRegressor
    model = ExtraTreesRegressor(random_state=0)
    model.fit(data,target)
    importance = model.feature_importances_

    sorted_feature = []
    indices = np.argsort(importance)[::-1]
    for f in range(data.shape[1]):
        classi = "%d. feature %s (%f)" % (f + 1, data.columns[indices[f]], importance[indices[f]])
        print(classi)
        sorted_feature.append(data.columns[indices[f]]) 
    
    # Plot the feature importances of the forest
    fig,ax = plt.subplots()
    plt.title("Feature importances with ExtraTreesClassifier",fontsize=15)
    if text==1:
        for i, v in enumerate(importance[indices]):
            ax.text(i - 0.15, v + 0.005, round(v,2),fontsize=15)
    
    plt.bar(range(data.shape[1]), importance[indices], color="r", align="center")
    plt.xticks(range(data.shape[1]), sorted_feature,rotation=90,fontsize=15)
    plt.xlim([-1, data.shape[1]])
    plt.show()

# Affichage de l'importance des variable grace à un arbre SelectKBest et chi2
def importance_features_k(data,target):
    selector = SelectKBest (chi2, k= 'all')
    X_new = selector.fit_transform(data,target)
    names = data.columns.values[selector.get_support()]
    scores = selector.scores_[selector.get_support()]

    names_scores = list(zip(names, scores))
    ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
    #Sort the dataframe for better visualization
    ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
    return ns_df_sorted


# # Intégration et compréhension des données

# In[104]:


data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
print("shape : ",data.shape)


# *Aperçu des données*

# In[105]:


data.head()


# *Types des données*

# In[106]:


# Type des données
print(data.dtypes)


# In[107]:


# Analyse de la variable Survived, qui sera notre target.
data.Survived.value_counts()


# ## Distribution des variables entières

# In[108]:


#Liste des variables entières : 
print(data.select_dtypes(exclude=['object']).columns)


# ### PassengerId

# In[109]:


# Analyse de la variable PassengerId
print("Il y a %s valeurs distinctes, pour %s données.Il y a donc un id unique pour chaque passager."
%(len(data.PassengerId.unique()),len(data.PassengerId)))


# ### PClass

# In[110]:


# Analyse de la variable PClass (Ticket class)
print(data.Pclass.value_counts())
print("Il y a %s valeur(s) manquante(s)." %data.Pclass.isnull().sum())
ax = sns.countplot(x="Pclass", data=data, palette="Set2")
plt.title('Distribution des PClass')
plt.show()


# In[111]:


# Analyse de la variable PClass par rapport à Survived)
g = sns.factorplot(x="Pclass",col="Survived", data=data,kind="count")
plt.show()


# ### Age

# In[112]:


# Analyse de la variable Age
print("Il y a %s valeur(s) manquante(s)." %data.Age.isnull().sum())
data.Age.value_counts(bins=15,sort=False).plot(kind='barh')
plt.title('Distribution des Ages')
plt.show()


# In[113]:


#Distribution des ages lorsque le passager a survecu et n'a pas survécu
df_age = pd.DataFrame(columns = ['Survived','Not Survived'])
df_age['Survived'] = data[data.Survived == 1].Age.describe()
df_age['Not Survived'] = data[data.Survived == 0].Age.describe()
df_age


# ### SibSp

# In[114]:


# Analyse de la variable SibSp (# de personne dans leur famille)
print(data.SibSp.value_counts())
print("Il y a %s valeur(s) manquante(s)." %data.SibSp.isnull().sum())
ax = sns.countplot(x="SibSp", data=data, palette="Set2")
plt.title('Distribution de SibSp')
plt.show()


# In[115]:


# Analyse de la variable SibSp par rapport à Survived)
g = sns.factorplot(x="SibSp",col="Survived", data=data,kind="count")
plt.show()


# ### Parch

# In[116]:


# Analyse de la variable Parch (# de parents ou d'enfants)
print(data.Parch.value_counts())
print("Il y a %s valeur(s) manquante(s)." %data.Parch.isnull().sum())
ax = sns.countplot(x="Parch", data=data, palette="Set2")
plt.title('Distribution de Parch')
plt.show()


# In[117]:


# Analyse de la variable Parch par rapport à Survived)
g = sns.factorplot(x="Parch",col="Survived", data=data,kind="count")
plt.show()


# ### Fare

# In[118]:


# Analyse de la variable Fare
print("Intervalle : ",min(data.Fare.unique()),max(data.Fare.unique()))
print("Il y a %s valeur(s) manquante(s)." %data.Fare.isnull().sum())
ax = sns.distplot(data.Fare)
plt.title('Distribution de Fare')
plt.show()


# ## Distribution des variables object

# In[119]:


#Liste des variables entières : 
print(data.select_dtypes(include=['object']).columns)


# ### Sex

# In[120]:


# Analyse de la variable Sex
print(data.Sex.value_counts())
print("Il y a %s valeur(s) manquante(s)." %data.Sex.isnull().sum())
ax = sns.countplot(x="Sex", data=data, palette="Set2")
plt.title('Distribution de sex')
plt.show()


# In[121]:


# Analyse de la variable Sex par rapport à Survived)
g = sns.factorplot(x="Sex",col="Survived", data=data,kind="count")
plt.show()


# ### Cabin

# In[122]:


# Analyse de la variable Cabin
print(data.Cabin.unique())
print("Il y a %s valeur(s) manquante(s)." %data.Cabin.isnull().sum())
data.shape


# ### Ticket

# In[123]:


# Analyse de la variable Ticket
print(data.Ticket.describe())
print("Il y a %s valeur(s) manquante(s)." %data.Ticket.isnull().sum())


# In[124]:


dfamily = pd.DataFrame(columns = ["Percentage Survived","Survived","No Survived",'Nb children','Nb adult','Nb Unknown','Nb Tot','PClass'])
i=0
while data.Ticket.value_counts().values[i] > 1: #tant que c'est une famille
    subset = data[data.Ticket == data.Ticket.value_counts().index[i]]
    if 1 not in subset.Survived.values:
        sur = 0
    else :
        sur = subset.Survived.value_counts().loc[1]
    if 0 not in subset.Survived.values:
        notsur = 0
    else :
        notsur = subset.Survived.value_counts().loc[0]
    child = subset[subset.Age <= 18].shape[0]
    unknown = subset[subset.Age.isnull()].shape[0]
    adult= subset.shape[0] - (child+unknown)
    clas = subset.Pclass.iloc[0] 
    per = (sur/(sur+notsur))*100
    tot = sur+notsur
    listi = [[per,sur,notsur,child,adult,unknown,tot,clas]]
    dfamily = dfamily.append(pd.DataFrame(listi, columns=dfamily.columns),ignore_index=True)
    i=i+1


# *On a ici créé un tableau qui contient pour chaque famille, des caractéristiques*

# In[125]:


dfamily.head()


# In[126]:


#Pourcentage de personnes qui ont survécu dans les familles
dfamily.groupby(['Nb Tot'])['Percentage Survived'].mean()


# ### Embarked

# In[127]:


# Analyse de la variable Embarked
print(data.Embarked.value_counts())
print("Il y a %s valeur(s) manquante(s)." %data.Embarked.isnull().sum())
ax = sns.countplot(x="Embarked", data=data, palette="Set2")
plt.title('Distribution de Embarked')
plt.show()


# In[128]:


# Analyse de la variable Embarked par rapport à Survived
g = sns.factorplot(x="Embarked",col="Survived", data=data,kind="count")
plt.show()


# # Gestion des valeurs manquantes 

# In[129]:


print_MV_percentage(data)


# In[130]:


print_MV_percentage(test)


# ## Cabin

# In[131]:


#Gestion pour Cabin
print("Il y a %.1f %% de valeurs manquantes dans les données d'entrainement" %(data.Cabin.isnull().sum()*100/len(data.Cabin)))


# On choisit donc de retirer cette variable car un remplissage de
# valeurs manquantes ne serait pas forcément bien représentatif
# de la réalité étant donné qu'il nous manque plus des trois quart des données.

# In[132]:


data = data.drop('Cabin', axis=1)
test = test.drop('Cabin', axis=1)


# ## Age

# In[133]:


#Gestion pour Age
print("Il y a %.1f %% de valeurs manquantes dans les données d'entrainement" %(data.Age.isnull().sum()*100/len(data.Age)))


# Ici, nous pouvons appliquer la méthode de remplissage de KNN.  
# La valeur à remplir est l'age, qui est un float.

# In[134]:


table = pd.DataFrame()
print(data[data.Pclass == 1].Age.mean())
print(data[data.Pclass == 2].Age.mean())
print(data[data.Pclass == 3].Age.mean())


# In[135]:


fill_MV_KNN(data)
fill_MV_KNN(test)


# ## Embarked

# In[136]:


#Gestion pour Embarked
print("Il y a %.1f %% de valeurs manquantes" %(data.Embarked.isnull().sum()*100/len(data.Embarked)))


# In[137]:


data.Embarked.fillna(data['Embarked'].mode()[0], inplace=True)


# On choisit de remplacer les deux valeurs manquantes par la porte d'embarquation qui sort le plus souvent.

# ## Fare

# *Il n'y a ici qu'une seule valeur manquante, un simple remplissage par la moyenne suffit.*

# In[138]:


test.Fare.fillna(test.Fare.mean(), inplace=True)


# Nous avons maintenant plus aucune valeurs manquantes !

# # Feature Engineering

# ## Catégories d'age

# In[139]:


# On crée une nouvelle colonne 'age_group' qui nous indiquera le groupe de l'individu
bins_age = [0, 16, 32, 48, 64, np.inf]
labels_age = [0,1,2,3,4]
Age_group = pd.cut(data.Age, bins_age, labels=labels_age)
data['Age_group'] = Age_group
Age_group_test = pd.cut(test.Age, bins_age, labels=labels_age)
test['Age_group'] = Age_group_test


# In[140]:


print(data.Age_group.value_counts())
ax = sns.countplot(x="Age_group", data=data, palette="Set2")
plt.title('Distribution de Age_group')
plt.show()


# ## Catégories de Fare

# In[141]:


# On effectue la mème chose avec Fare.
bins_fare = [-np.inf, 8, 15, 31, np.inf]
labels_fare = [1,2,3,4]
Fare_group = pd.cut(data.Fare, bins_fare, labels=labels_fare)
data['Fare_group'] = Fare_group
Fare_group_test = pd.cut(test.Fare, bins_fare, labels=labels_fare)
test['Fare_group'] = Fare_group_test


# In[142]:


print(data.Fare_group.value_counts())
ax = sns.countplot(x="Fare_group", data=data, palette="Set2")
plt.title('Distribution de Fare_group')
plt.show()


# ## Taille de la famille

# In[143]:


# Création d'une colonne qui désigne la taille de la famille
FamilySize = data.SibSp + data.Parch + 1
data['FamilySize'] = FamilySize
data['IsAlone']=0
data.loc[data['FamilySize'] == 1, 'IsAlone'] = 1

FamilySize_test = test.SibSp + test.Parch + 1
test['FamilySize'] = FamilySize_test
test['IsAlone']=0
test.loc[test['FamilySize'] == 1, 'IsAlone'] = 1


# In[144]:


data.head()


# # Conversion ou Suppression de colonnes

# Tout d'abord, on remarque que les colonnes PassengerId et Ticket ne seront pas intéressante pour la construction de notre modèle.  
# En effet, elle sont distribué de façon très aléatoire et ne seront donc pas représentative du phénomène.

# In[145]:


# Suppression de PassengerId et Ticket
data = data.drop(['PassengerId','Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
data.head()


# ## Mapping de Name

# In[146]:


# On décide de ne prendre que le titre de la colonne Name 
#(car le nom n'est pas représentatif du phénomène)
normal = ["Mr","Miss","Mrs","Master"]
title=[]
for row in data.Name:
    row = row.replace('.',',').replace(' ','')
    t = row.split(',')[1]
    if t in normal :
        title.append(row.split(',')[1])
    else : 
        title.append('aris')
Titles = pd.Series(title)
data['Title'] = Titles

title=[]
for row in test.Name:
    row = row.replace('.',',').replace(' ','')
    t = row.split(',')[1]
    if t in normal :
        title.append(row.split(',')[1])
    else : 
        title.append('aris')
Titles = pd.Series(title)
test['Title'] = Titles

data.head()


# In[147]:


data = data.drop(['Name'],axis=1)
test = test.drop(['Name'],axis=1)


# ## Encoding

# In[148]:


data_encode = encoding(data)
test_encode = encoding(test)


# In[149]:


data_one = pd.get_dummies(data)
test_one = pd.get_dummies(test)


# ## Création d'une colonne de Cluster

# In[150]:


from sklearn.preprocessing import StandardScaler
data_scale = StandardScaler().fit_transform(data_encode.drop(['Survived'], axis=1))
test_scale = StandardScaler().fit_transform(test_encode)


# In[151]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
list_sil = []
list_sil_test = []
for n_clusters in range(2,20):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(data_scale)
    clusters = kmeans.predict(data_scale)
    silhouette_avg = silhouette_score(data_scale, clusters)
    list_sil.append(silhouette_avg)
    
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(test_scale)
    clusters = kmeans.predict(test_scale)
    silhouette_avg = silhouette_score(test_scale, clusters)
    list_sil_test.append(silhouette_avg)


plt.subplot(211) 
plt.title("Variation du score du silhouette en fonction du nombre de cluster avec les données ENCODE")
plt.xticks(np.arange(2,20))
plt.plot(np.arange(2,20),list_sil)

plt.subplot(212) 
plt.title("TEST - Variation du score du silhouette en fonction du nombre de cluster avec les données ENCODE")
plt.xticks(np.arange(2,20))
plt.plot(np.arange(2,20),list_sil_test)

plt.tight_layout()
plt.show()


# In[152]:


from sklearn.preprocessing import StandardScaler
data_scale_one = StandardScaler().fit_transform(data_one.drop(['Survived'], axis=1))
test_scale_one = StandardScaler().fit_transform(test_one)


# In[153]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
list_sil = []
list_sil_test = []
for n_clusters in range(2,20):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(data_scale_one)
    clusters = kmeans.predict(data_scale_one)
    silhouette_avg = silhouette_score(data_scale_one, clusters)
    list_sil.append(silhouette_avg)
    
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(test_scale_one)
    clusters = kmeans.predict(test_scale_one)
    silhouette_avg = silhouette_score(test_scale_one, clusters)
    list_sil_test.append(silhouette_avg)


plt.subplot(211) 
plt.title("Variation du score du silhouette en fonction du nombre de cluster avec les données ONE HOT ENCODE")
plt.xticks(np.arange(2,20))
plt.plot(np.arange(2,20),list_sil)

plt.subplot(212) 
plt.title("TEST - Variation du score du silhouette en fonction du nombre de cluster avec les données ONE HOT ENCODE")
plt.xticks(np.arange(2,20))
plt.plot(np.arange(2,20),list_sil_test)

plt.tight_layout()
plt.show()


# In[154]:


#ENCODE SIMPLE
from sklearn.cluster import KMeans

km_data = KMeans(n_clusters=9).fit(data_scale)
km_test = KMeans(n_clusters=9).fit(test_scale)

cluster_map_data = pd.DataFrame()
cluster_map_data['data_index'] = data.index.values
cluster_map_data['cluster'] = km_data.labels_

cluster_map_test = pd.DataFrame()
cluster_map_test['data_index'] = test.index.values
cluster_map_test['cluster'] = km_test.labels_

data_encode['Cluster'] = cluster_map_data['cluster']
test_encode['Cluster'] = cluster_map_test['cluster']
data_encode.head()


# In[155]:


#ENCODE ONE HOT
km_data = KMeans(n_clusters=12).fit(data_scale_one)
km_test = KMeans(n_clusters=12).fit(test_scale_one)

cluster_map_data = pd.DataFrame()
cluster_map_data['data_index'] = data.index.values
cluster_map_data['cluster'] = km_data.labels_

cluster_map_test = pd.DataFrame()
cluster_map_test['data_index'] = test.index.values
cluster_map_test['cluster'] = km_test.labels_

data_one['Cluster'] = cluster_map_data['cluster']
test_one['Cluster'] = cluster_map_test['cluster']
data_one.head()


# # Features Importances

# *Encoding simple*

# In[156]:


from sklearn.ensemble import ExtraTreesClassifier
target = data_encode.Survived
preds = data_encode.drop(['Survived'],axis=1)


# In[157]:


feature_importance (preds,target,1)


# In[163]:


importance_features_k(preds,target)


# *One-hot Encoding*

# In[164]:


target_one = data_one.Survived
preds_one = data_one.drop(['Survived'],axis=1)


# In[165]:


feature_importance (preds_one,target_one,0)


# In[166]:


importance_features_k(preds_one,target_one)


# # Machine Learning

# In[167]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import copy


xtrain, xtest, ytrain, ytest = train_test_split(data_encode, target, random_state=0)

xtest_annoted = copy.deepcopy(xtest)
xtrain = xtrain.drop(['Survived'], axis=1)
xtest = xtest.drop(['Survived'], axis=1)

# Définition du model
forest_model = RandomForestClassifier()
# Entrainement du modèle
forest_model.fit(xtrain, ytrain)
pred = forest_model.predict(xtest)

# On met les deux colonnes à coté
xtest_annoted['Real']=xtest_annoted['Survived']
xtest_annoted = xtest_annoted.drop(['Survived'], axis=1)
xtest_annoted['Predicted']=pred

# Création d'une df qui regroupe les mauvaises prédictions
x_false = xtest_annoted[xtest_annoted.Real != xtest_annoted.Predicted]


print(round(forest_model.score(xtest, ytest)*100,2))
print(confusion_matrix(pred, ytest))
mean_absolute_error(pred, ytest)


# In[168]:


from sklearn.model_selection import cross_val_score

X = data_one.drop(['Survived'], axis=1)
y = copy.deepcopy(target)

scores = cross_val_score(forest_model, X, y)
print(scores.mean()*100)


# In[169]:


print(x_false.shape)
x_false.head()


# In[170]:


plt.subplot(3,2, 1)
ax = sns.countplot(x="Pclass", data=x_false, palette="Set2")
plt.subplot(3,2, 2)
ax = sns.countplot(x="Sex", data=x_false, palette="Set2")
plt.subplot(3,2, 3)
ax = sns.countplot(x="Fare_group", data=x_false, palette="Set2")
plt.subplot(3,2, 4)
ax = sns.countplot(x="Title", data=x_false, palette="Set2")
plt.subplot(3,2, 5)
ax = sns.countplot(x="Cluster", data=x_false, palette="Set2")
plt.show()


# In[171]:


print(test.shape)
pred = forest_model.predict(test_encode.drop(['PassengerId'], axis=1))
submission = pd.DataFrame({"PassengerId": test.PassengerId,"Survived": pred})
submission.to_csv('submission.csv', index=False)


# Avec ce résultat on obtient un score de 80 %

# ## Essai avec Pipeline

# In[172]:


list_score = []
list_scoreO = []

xtrain, xtest, ytrain, ytest = train_test_split(data_encode, target, random_state =0)
xtrainO, xtestO, ytrainO, ytestO = train_test_split(data_one, target, random_state =0)


# ### Random Forest

# Pour ce modèle, on essaie de trouver les meilleurs paramètre grace à la fonction GridSearchCV

# In[173]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

xtrain = xtrain.drop(['Survived'], axis=1)
xtest = xtest.drop(['Survived'], axis=1)

forest_model = RandomForestClassifier()
param_grid = {
    'bootstrap': [True],
    'max_depth': [110],
    'min_samples_leaf': [3,2],
    'min_samples_split': [10,2],
    'n_estimators': [200]
}
grid = GridSearchCV(estimator=forest_model, param_grid=param_grid)
grid.fit(xtrain,ytrain)
print(grid.best_score_)
print(grid.best_estimator_)

my_pipeline_RF = make_pipeline(grid.best_estimator_)
my_pipeline_RF.fit(xtrain, ytrain)
predictions = my_pipeline_RF.predict(xtest)

list_score.append(1-mean_absolute_error(predictions, ytest))


# In[174]:


xtrainO = xtrainO.drop(['Survived'], axis=1)
xtestO = xtestO.drop(['Survived'], axis=1)

forest_model = RandomForestClassifier()
param_grid = {
    'bootstrap': [True],
    'max_depth': [110],
    'min_samples_leaf': [3,2],
    'min_samples_split': [10,2],
    'n_estimators': [200]
}
grid = GridSearchCV(estimator=forest_model, param_grid=param_grid)
grid.fit(xtrainO,ytrainO)
print(grid.best_score_)
print(grid.best_estimator_)

my_pipeline_RF = make_pipeline(grid.best_estimator_)
my_pipeline_RF.fit(xtrainO, ytrainO)
predictions = my_pipeline_RF.predict(xtestO)

list_scoreO.append(1-mean_absolute_error(predictions, ytestO))


# ### KNN

# In[175]:


xtrain, xtest, ytrain, ytest = train_test_split(data_scale, target, random_state =0)
xtrainO, xtestO, ytrainO, ytestO = train_test_split(data_scale_one, target, random_state =0)


# In[176]:


from sklearn.neighbors import KNeighborsClassifier
errors=[]
for k in range (2,25):
    knn = KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest,ytest)))
plt.xlabel("valeur de k")
plt.ylabel("valeur de l'erreur")
plt.plot(range(2,25),errors, 'o-')
plt.show()
mini_k=errors.index(min(errors))+2


# In[177]:


errors=[]
for k in range (2,25):
    knn = KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrainO, ytrainO).score(xtestO,ytestO)))
plt.xlabel("valeur de k")
plt.ylabel("valeur de l'erreur")
plt.plot(range(2,25),errors, 'o-')
plt.show()
mini_kO=errors.index(min(errors))+2


# In[178]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_absolute_error

my_pipeline_KNN = make_pipeline(Imputer(),KNeighborsClassifier(n_neighbors=mini_k))
my_pipeline_KNN.fit(xtrain, ytrain)
predictions = my_pipeline_KNN.predict(xtest)

list_score.append(1-mean_absolute_error(predictions, ytest))
mean_absolute_error(predictions, ytest)


# In[179]:


my_pipeline_KNN = make_pipeline(Imputer(),KNeighborsClassifier(n_neighbors=mini_kO))
my_pipeline_KNN.fit(xtrainO, ytrainO)
predictions = my_pipeline_KNN.predict(xtestO)

list_scoreO.append(1-mean_absolute_error(predictions, ytestO))
mean_absolute_error(predictions, ytestO)


# ### Support Vector Machine

# In[180]:


xtrain, xtest, ytrain, ytest = train_test_split(data_encode, target, random_state =0)
xtrainO, xtestO, ytrainO, ytestO = train_test_split(data_one, target, random_state =0)

xtrain = xtrain.drop(['Survived'], axis=1)
xtest = xtest.drop(['Survived'], axis=1)

xtrainO = xtrainO.drop(['Survived'], axis=1)
xtestO = xtestO.drop(['Survived'], axis=1)


# In[181]:


from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

my_pipeline_SVC = make_pipeline(SVC())
my_pipeline_SVC.fit(xtrain, ytrain)
predictions = my_pipeline_SVC.predict(xtest)

list_score.append(1-mean_absolute_error(predictions, ytest))


# In[182]:


my_pipeline_SVC = make_pipeline(SVC())
my_pipeline_SVC.fit(xtrainO, ytrainO)
predictions = my_pipeline_SVC.predict(xtestO)

list_scoreO.append(1-mean_absolute_error(predictions, ytestO))


# ### Decision Tree

# In[183]:


from sklearn.tree import DecisionTreeClassifier

my_pipeline_DT = make_pipeline(DecisionTreeClassifier())
my_pipeline_DT.fit(xtrain, ytrain)
predictions = my_pipeline_DT.predict(xtest)

list_score.append(1-mean_absolute_error(predictions, ytest))


# In[184]:


my_pipeline_DT = make_pipeline(DecisionTreeClassifier())
my_pipeline_DT.fit(xtrainO, ytrainO)
predictions = my_pipeline_DT.predict(xtestO)

list_scoreO.append(1-mean_absolute_error(predictions, ytestO))


# ### Logistic Regression

# In[185]:


from sklearn.linear_model import LogisticRegression

my_pipeline_LR = make_pipeline(LogisticRegression())
my_pipeline_LR.fit(xtrain, ytrain)
predictions = my_pipeline_LR.predict(xtest)

list_score.append(1-mean_absolute_error(predictions, ytest))


# In[186]:


my_pipeline_LR = make_pipeline(LogisticRegression())
my_pipeline_LR.fit(xtrainO, ytrainO)
predictions = my_pipeline_LR.predict(xtestO)

list_scoreO.append(1-mean_absolute_error(predictions, ytestO))


# ### On regarde les scores de tous les modèles

# In[187]:


models = pd.DataFrame({
    'Model': ['Random Forest', 'KNN', 'Support Vector Machines','Decision Tree','Linear Regression'],
    'Score': list_score,
    'Score One Hot': list_scoreO})
models.sort_values(by='Score', ascending=False)


# In[188]:


pred = my_pipeline_RF.predict(test_one.drop(['PassengerId'], axis=1))
submission = pd.DataFrame({"PassengerId": test.PassengerId,"Survived": pred})
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




