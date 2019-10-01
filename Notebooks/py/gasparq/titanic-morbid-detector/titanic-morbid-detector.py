#!/usr/bin/env python
# coding: utf-8

# # Problème
# 
# Notre but est de prédire quels passagers du Titanic vont survivre ou mourir au naufrage.
# 
# Pour se faire, nous avons accès aux données suivantes :
# * Classe du siège (1ère, Business, Eco)
# * Nom du passager
# * Sexe du passager
# * Age du passager
# * Nombre de frère/soeur et maris/épouse du passager
# * Nombre d'enfants et parents du passager
# * Numéro du ticket
# * Frais engagés
# * Numéro de Cabine (incomplet)
# * Port d'embarquement

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as sk
from tensorflow import keras
import os
import seaborn as sns
import sklearn.ensemble as skl

print(os.listdir("../input"))

full_train_set = pd.read_csv('../input/train.csv')
final_set = pd.read_csv('../input/test.csv')

print('Train set: ', full_train_set.shape)
print('Final set: ', final_set.shape)


# # Hypothèses
# 
# ## Introduction
# 
# Intuitivement, on peut essayer de dégager l'importance de chacune des données fournies.
# 
# Le **nom du passager** n'aura pas d'importance pour dégager un quelconque modèle à suivre.
# 
# Ensuite, le **sexe** et **l'âge** du passager auront une grande importance car : "les femmes et les enfants d'abords".
# 
# La **classe du siège** ainsi que les **frais engagés** vont influer car les gens les plus fortunés auront la priorité d'accès aux canaux de sauvetage (suposition).
# 
# Le nombre de **frère/soeur**, **maris/épouse**, **enfants/parents** aura une certaine importance car les familles s'entraideront pour s'en sortir tous ensemble.
# 
# Le **numéro du ticket** a une certaine importance dans la mesure où on peut penser aux personnes trop proches de l'impact pour remonter à la surface. Il est néanmoins inutilisable tels quel.
# 
# Le **numéro de cabine** est dans l'état peu utilisable vu qu'il manque beaucoup de données.
# 
# Le **port d'embarquement** n'a pas spécialement d'importance car celui-ci s'est fait dans la journée du 10 avril 1992 donc tous les passager ont à passer le même temps sur le bâteau (à quelques heures près).
# 
# Afin de vérifier ces suppositions, nous allons explorer les données

# ## Les femmes et les enfants ont plus survécu
# 
# Selon la phrase "les femmes et les enfants d'abords", on pourrait s'attendre à ce que les femmes et les enfants aient plus survécus que les hommes.

# In[ ]:


adults = full_train_set.loc[full_train_set['Age'] >= 18]
children = full_train_set.loc[full_train_set['Age'] < 18]
women = adults.loc[full_train_set['Sex'] == 'female']
men = adults.loc[full_train_set['Sex'] == 'male']

children_alive = children.loc[full_train_set['Survived'] == 1].shape[0]
children_dead = children.loc[full_train_set['Survived'] == 0].shape[0]

women_alive = women.loc[full_train_set['Survived'] == 1].shape[0]
women_dead = women.loc[full_train_set['Survived'] == 0].shape[0]

men_alive = men.loc[full_train_set['Survived'] == 1].shape[0]
men_dead = men.loc[full_train_set['Survived'] == 0].shape[0]

def print_stats(data_alive, data_dead, name):
    print(name, ' Alive => ', data_alive, ' Dead => ', data_dead, ' Ratio alive => ', (data_alive / (data_alive + data_dead)) * 100, '%')

print_stats(men_alive, men_dead, 'Men')
print_stats(women_alive, women_dead, 'Men')
print_stats(children_alive, children_dead, 'Men')

ind = np.arange(3)
alive_data = [children_alive, women_alive, men_alive]
dead_data = [children_dead, women_dead, men_dead]

alives = plt.bar(ind, alive_data)
deads = plt.bar(ind, dead_data, color='#d62728', bottom=alive_data)

plt.ylabel('Nombre')
plt.title('Proportion des morts par sexe')
plt.xticks(ind, ('children', 'women', 'men'))
plt.legend((alives, deads), ('Alive', 'Dead'))


# Selon les observations, on remarque que les femmes ont survécu en plus grande proportion que les hommes, cependant, le taux de survie des enfants est à peu près moitié-moitié.
# 
# Mais l'intuition était bonne, les hommes ont un ratio de survie bien inférieur aux autres, l'information est donc pertinente.

# ## Les gens plus fortunés ont plus survécu
# 
# Selon la loi du plus fort transposé sur un système capitaliste, on pourrait s'attendre à ce que les gens les plus puissants, à savoir les gens fortunés, aient plus survécu que les plus faibles, à savoir les pauvres.
# 
# Nous allons donc évaluer cette théorie sur nos données.

# In[ ]:


first_class = full_train_set.loc[full_train_set['Pclass'] == 1]
second_class = full_train_set.loc[full_train_set['Pclass'] == 2]
third_class = full_train_set.loc[full_train_set['Pclass'] == 3]

dead_first_class, alive_first_class = first_class.loc[first_class['Survived'] == 0].shape[0], first_class.loc[first_class['Survived'] == 1].shape[0]
dead_second_class, alive_second_class = second_class.loc[second_class['Survived'] == 0].shape[0], second_class.loc[second_class['Survived'] == 1].shape[0]
dead_third_class, alive_third_class = third_class.loc[third_class['Survived'] == 0].shape[0], third_class.loc[third_class['Survived'] == 1].shape[0]

max_fare, fare_step = 50, 5
fare_peoples = [np.zeros(int(max_fare / fare_step) + 1), np.zeros(int(max_fare / fare_step) + 1)]

for i in range(0, max_fare + 1, fare_step):
    index = int(i / fare_step)
    curr_fares = full_train_set.loc[(full_train_set['Fare'] > i) & (full_train_set['Fare'] <= i + fare_step)]
    fare_peoples[0][index] = curr_fares.loc[curr_fares['Survived'] == 0].shape[0]
    fare_peoples[1][index] = curr_fares.loc[curr_fares['Survived'] == 1].shape[0]

plt.figure(2, figsize=(20, 20))

ind = ('1st class', '2nd class', '3rd class')
alive_data = [alive_first_class, alive_second_class, alive_third_class]
dead_data = [dead_first_class, dead_second_class, dead_third_class]

print(ind)
print('Alive', alive_data)
print('Dead', dead_data)
print()

plt.subplot(221)
alive = plt.bar(ind, alive_data)
dead = plt.bar(ind, dead_data, bottom=alive_data)

plt.ylabel('Nombre')
plt.title('Proportion de mort par classe')
plt.legend((alive, dead), ('Alive', 'Dead'))

ind = np.arange(0, max_fare + 1, fare_step)
print('Fares', ind)
print('Alive', fare_peoples[1])
print('Dead', fare_peoples[0])

plt.subplot(222)
alive = plt.bar(ind, fare_peoples[1], width=3)
dead = plt.bar(ind, fare_peoples[0], bottom=fare_peoples[1], width=3)

plt.ylabel('Survivant')
plt.xlabel('Frais engagés')
plt.title('Proportion de morts par frais')
plt.xticks(ind, ['{}-{}£'.format(i, i + fare_step) for i in ind])
plt.legend((alive, dead), ('Alive', 'Dead'))


# Selon les observations, il semblerait que les passagers en 3ème classes et ceux investissant le moins de frais soient ceux qui aient le moins survécu.
# 
# Notre hypothèse était donc vraie à ce propos et ces données sont donc importantes pour la détermination du résultat.

# ## Les familles s'entraident
# 
# Ont sais que certaines personnes sont venues avec leur parents, enfants, frères, soeur, maris et épouses.
# 
# Etant donné que chacun des membres se connais, il semblerait normal que les uns veillent sur les autres.
# 
# Mais il est possible que cela fasse un effet totalement imprévu et chaotique si des personnes essaient désespérément de sauver des membres de leur familles.

# In[ ]:


parch_peoples = np.zeros((7,2))
sibsp_peoples = np.zeros((9,2))
full_peoples = np.zeros((16,2))
groupped = np.zeros((2, 2))

for row in full_train_set.iterrows():
    passenger = row[1]
    alive = passenger['Survived']
    parch = passenger['Parch']
    sibsp = passenger['SibSp']
    family_size = parch + sibsp
    alone = 1 if family_size != 0 else 0
    
    parch_peoples[parch, alive] += 1
    sibsp_peoples[sibsp, alive] += 1
    full_peoples[family_size, alive] += 1    
    groupped[alone] += 1
    
def print_data(ind, data, splot, title):
    adata = data[:,1:].reshape(-1)
    ddata = data[:,:1].reshape(-1)
    print('==========', title, '==========')
    print('Alive:', adata)
    print('Dead:', ddata)
    print('==========' + ('=' * (len(title) + 2)) + '==========')
    plt.subplot(splot)
    alive = plt.bar(ind, adata)
    dead = plt.bar(ind, ddata, bottom=adata)
    plt.legend((alive, dead), ('Alive', 'Dead'))
    plt.ylabel('Passager')
    plt.xlabel('Nombre membres famille')
    plt.title(title)

plt.figure(4, figsize=(10,10))

print_data(np.arange(7), parch_peoples, 221, 'Proportion de morts avec parents/enfants')
print_data(np.arange(9), sibsp_peoples, 222, 'Proportion de morts avec frère/soeur maris/épouse')
print_data(np.arange(16), full_peoples, 223, 'Proportion de morts totale avec famille')
print_data(np.arange(2), groupped, 224, 'Proportion de morts avec/sans famille')
plt.xticks((0, 1), ("Alone", "Group"))
plt.xlabel('')


# Au vue des observations, on ne peut conclure que la famille ai un grand impact dans la survie ou la mort des passagers.
# 
# En effet, le personnes en ayant ont un ratio de survie d'enrivon 50% tandis que les autres d'environ 30% mais le nombre de personne sans famille et bien trop élevé par rapport au reste pour que les ratio restants pèsent dans la balance.
# 
# Nous ferons donc plusieurs tests avec cette donnée pour vérifier qu'elle possède bien une importance dans le résultat final.

# # Sélection des données
# 
# Au vue des observations, nous allons tout d'abords prendre en compte le données :
# * Le sexe
# * L'âge
# * La classe
# * Les frais
# * Famille
# 
# Nous allons donc reformater les données à envoyer

# In[ ]:


"""
full_labels = full_train_set["Survived"]
"""
full_labels = pd.DataFrame({
    "IsDead": [1 - survived for survived in full_train_set["Survived"]],
    "IsAlive": full_train_set["Survived"]
})

def CheckIfAlone(passenger):
    parch = passenger['Parch']
    sibsp = passenger['SibSp']
    family_size = parch + sibsp
    return 1 if family_size != 0 else 0

def GetInputModel(features):
    return pd.DataFrame({
        "IsMan": [(1 if sex == 'male' else 0) for sex in features["Sex"]],
        "IsWomen": [(1 if sex == 'female' else 0) for sex in features["Sex"]],
        "IsChild": [(1 if age < 18 else 0) for age in features["Age"]],
        "IsFclass": [(1 if pclass == 1 else 0) for pclass in features["Pclass"]],
        "IsSclass": [(1 if pclass == 2 else 0) for pclass in features["Pclass"]],
        "IsTclass": [(1 if pclass == 3 else 0) for pclass in features["Pclass"]],
        "Fare1": [(1 if fare > 5 and fare <= 10 else 0) for fare in features["Fare"]],
        "Fare2": [(1 if fare > 10 and fare <= 15 else 0) for fare in features["Fare"]],
        "Fare3": [(1 if fare > 15 and fare <= 20 else 0) for fare in features["Fare"]],
        "Fare4": [(1 if fare > 20 else 0) for fare in features["Fare"]],
        "IsAlone": [CheckIfAlone(row[1]) for row in features.iterrows()],
        "IsGroup": [1 - CheckIfAlone(row[1]) for row in features.iterrows()]
    })

full_features = GetInputModel(full_train_set)
final_features = GetInputModel(final_set)


print('Labels: ', full_labels.shape)
print('Features: ', full_features.shape)

train_labels, test_labels, train_features, test_features = sk.train_test_split(full_labels, full_features, train_size=0.75)

print('Train labels: ', train_labels.shape, ' | Train features: ', train_features.shape)
print('Test labels: ', test_labels.shape, ' | Test features: ', test_features.shape)


# # Construction du modèle
# 
# Nous allons ensuite construire notre modèle avec keras.

# In[ ]:


model = keras.Sequential([
    keras.layers.Dense(12, input_shape=(12,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation=keras.activations.softmax)
])

model.compile(keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.fit(train_features, train_labels, batch_size=32, epochs=500)

los, acc = model.evaluate(test_features, test_labels, batch_size=32)

print('Loss: ', los, ' | Accuracy: ', acc)


# In[ ]:


model = skl.RandomForestClassifier(n_estimators=1000, max_depth=500, random_state=1)

model.fit(train_features, train_labels)

train_score = model.score(train_features, train_labels)
test_score = model.score(test_features, test_labels)

print('Train score: ', train_score, ' Test score: ', test_score)

features_imp = pd.DataFrame({
    "Names": train_features.columns,
    "Values": model.feature_importances_
})

print(features_imp)

sns.set_color_codes("pastel")
sns.barplot(x="Values", y="Names", data=features_imp,
            label="Importance des colonnes", color="b")


# # Soumission du kernel
# 
# Maintenant que nous avons des résultats assez satisfaisant avec notre modèle, nous allons l'appliquer à nos données finales.

# In[ ]:


final_labels = model.predict(final_features)

data = pd.DataFrame({
    "PassengerId": final_set["PassengerId"],
    "Survived": np.argmax(final_labels, axis=1)
})

print('Data ', data.shape)

data.to_csv("sumbission.csv", index=False)

