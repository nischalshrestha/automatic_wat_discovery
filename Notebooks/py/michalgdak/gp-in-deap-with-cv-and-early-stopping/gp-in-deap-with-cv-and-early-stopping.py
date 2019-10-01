#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division

import operator
import math
import random
import itertools
import pprint

import numpy as np
import matplotlib.pyplot as plt
from numpy import sort
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin


# In[ ]:


df = pd.read_csv("../input/trainbornclean1/train_born_clean.csv", sep=",")
df_test = pd.read_csv("../input/testbornclean2/test_born_clean.csv", sep=",")


# In[ ]:


def cleanData(dataset):
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna("S", inplace = True)

    #complete Cabin with mode
    dataset['Cabin'].fillna("0")
    
    #complete Born with mode
    dataset['Born'].fillna("Other", inplace = True)
    
    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace = True)
    
#fill NaN
cleanData(df)# = df.fillna(value={"Age": 0, "Born": "None"})
cleanData(df_test)# = df_test.fillna(value={"Age": 0, "Fare": 0, "Born": "None"})

#Convert dtype to correct format
coltypes = {"Pclass":"category", "Name": "str", "Sex":"category", "Age":"float",
        "SibSp":"int", "Parch":"int", "Ticket":"str", "Fare":"float", "Cabin":"str", "Embarked":"category", "Born": "category"}

for col, coltype in coltypes.items():
    df[col] = df[col].astype(coltype)

for col, coltype in coltypes.items():
    df_test[col] = df_test[col].astype(coltype)


# In[ ]:


display(df['Born'].mode())
display(df.isnull().sum())


# In[ ]:


def getFaimlyCount2(x):
    return x["SibSp"] + x["Parch"] + 1

def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'

def createFeatures(inDF):
    
    #deck
    inDF["Deck"] = inDF["Cabin"].apply(lambda x : x[:1])
    #toddler, kid, youth, adult, elderly pd.cut(inDF['Age'].astype(int), 5)#
    inDF["Maturity"] = pd.qcut(inDF['Age'].astype(int), 4)#inDF["Age"].apply(lambda x : "Unknown" if x==0 else "toddler" if x<2.0 else "kid" if x<10.0 else "youth" if x<18.0 else "adult" if x < 65.0 else "elderly")

    #Faimly size
    df_fs = inDF.groupby("Ticket").size().to_dict()
    inDF["#_of_per_ticket"] = inDF["Ticket"].apply(lambda x : df_fs[x])

    #Miss, Mrs, Mr, Master, ...
    inDF["Seniority"] = inDF["Name"].apply(lambda x : "Mr" if x.find("Mr. ") > 0 else "Mrs" if x.find("Mrs. ") > 0 else "Miss" if x.find("Miss. ") > 0 else "Master" if x.find("Master. ") > 0 else "None")
    
    #Surname
    inDF["Surname"] = inDF["Name"].apply(lambda x: x[0:x.find(",")])
    surname_size = inDF.groupby("Surname").size().to_dict()
    inDF["Surname"] = inDF["Surname"].apply(lambda x : "Other" if surname_size[x] <= 3 else x)
    
    #Family size
    inDF['family_size'] = inDF.apply(getFaimlyCount2, axis=1)
    inDF['family_size'] = inDF['family_size'].apply(lambda x : "Single" if x<=1 else "Double" if x<=2 else "MidF" if x<=4 else "LargeF")

    #pd.qcut(inDF['Fare'], 4)#
    inDF["Fare_type"] =pd.qcut(inDF['Fare'], 5)# inDF["Fare"].apply(lambda x : "Cheap" if x < 10 else "Mid" if x < 100 else "Exp")
    
    inDF['Ticket_type'] = inDF['Ticket'].apply(lambda x : cleanTicket(x))
    inDF['Cabin_type'] = inDF['Cabin'].apply(lambda x : x[:1])
    
    tfidf_vec = TfidfVectorizer(max_features=15, token_pattern="\w+")
    svd = TruncatedSVD(n_components=5)
    tfidf_array = svd.fit_transform(tfidf_vec.fit_transform(inDF["Name"]))
    for i in range(tfidf_array.shape[1]):
        inDF.insert(len(inDF.columns), column = 'tfidf_' + str(i), value = tfidf_array [:,i])

    born_dict = inDF.groupby("Born").size().to_dict()
    inDF["Born"]= inDF["Born"].apply(lambda x : "Other" if born_dict[x] < 10 else x)

df_all_before = df.append(df_test)

createFeatures(df_all_before)
#createFeatures(df_test)


# In[ ]:


train_data_col_to_use = ["Pclass", "Sex", "Fare_type", "Embarked", "Survived", "Maturity", "family_size",
                         "tfidf_0", "tfidf_1", "tfidf_2", "tfidf_3", "tfidf_4"]
                    #["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Born", "Survived", "Seniority", "Maturity",
                        #"family_size", "Fare_type", "tfidf_0", "tfidf_1", "tfidf_2", "tfidf_3", "tfidf_4", "Ticket_type"]

df_all = pd.get_dummies(df_all_before[train_data_col_to_use])

x = df_all.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_all = pd.DataFrame(x_scaled, columns=df_all.columns)

X_train = df_all[:891]
Y_train = X_train["Survived"].values
X_train = X_train.drop(["Survived"], axis=1)
X_pred = df_all[891:].drop(["Survived"], axis=1)


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.8, test_size=0.2)


# In[ ]:


points_for_GP = X_train.copy()
points_for_GP.insert(0, "Survived", Y_train)
points_for_GP.astype(float)

points_for_GP_CV = X_test.copy()
points_for_GP_CV.insert(0, "Survived", Y_test)
points_for_GP_CV.astype(float)

display(X_train.head(5))
display(points_for_GP.head(5))
display(points_for_GP.shape[1])


# In[ ]:


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def or_(left, right):
    return left | right
    
def and_(left, right):
    return left & right
    
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, points_for_GP.shape[1] - 1), bool)

# boolean operators
pset.addPrimitive(and_, [bool, bool], bool)
pset.addPrimitive(or_, [bool, bool], bool)
pset.addPrimitive(operator.not_, [bool], bool)

# floating point operators
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(protectedDiv, [float,float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(min, [float,float], float)
pset.addPrimitive(max, [float,float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin, [float], float)
pset.addPrimitive(math.tanh, [float], float)

# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2

pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand241", lambda: random.random(), float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)

pset.renameArguments(ARG0='Pclass')
pset.renameArguments(ARG1='Sex')
pset.renameArguments(ARG2='Age')
pset.renameArguments(ARG3='SibSp')
pset.renameArguments(ARG4='Parch')
pset.renameArguments(ARG5='Fare')
pset.renameArguments(ARG6='Embarked')
pset.renameArguments(ARG7="Maturity")
pset.renameArguments(ARG8="family_size")
pset.renameArguments(ARG9='tfidf_0')
pset.renameArguments(ARG10='tfidf_1')
pset.renameArguments(ARG11='tfidf_2')
pset.renameArguments(ARG12='tfidf_3')
pset.renameArguments(ARG13='tfidf_4')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, train, val):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)

    predictionsTrain = []  
    for index, point in train.iterrows():
        try:
            pred = bool(func(*point[1:]))#["Pclass"], point["Sex"], point["Age"], point["SibSp"], point["Parch"], point["Fare"],
                    #point["Embarked"], point["tfidf_0"], point["tfidf_1"], point["tfidf_2"], point["tfidf_3"], point["tfidf_4"]))
        except ValueError:
            pred = 0
        predictionsTrain.append(pred)
    
    predictionsVal = []  
    for index, point in val.iterrows():
        try:
            pred = bool(func(*point[1:]))#["Pclass"], point["Sex"], point["Age"], point["SibSp"], point["Parch"], point["Fare"],
                    #point["Embarked"], point["tfidf_0"], point["tfidf_1"], point["tfidf_2"], point["tfidf_3"], point["tfidf_4"]))
        except ValueError:
            pred = 0
        predictionsVal.append(pred)
    
    return log_loss(train["Survived"], predictionsTrain), log_loss(val["Survived"], predictionsVal), accuracy_score(val["Survived"], predictionsVal),

toolbox.register("evaluate", evalSymbReg, train=points_for_GP, val=points_for_GP_CV)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))


# In[ ]:



stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

population = toolbox.population(n=300)
cxpb, mutpb, ngen, mu, lambda_ = 0.6, 0.15, 300, 200, 400
stats = mstats
halloffame = tools.HallOfFame(1) 
verbose = True
wholeFitness = []
singleFitness = []
earlyStoppingThresh = 0.01
earlyStoppingGens = 150
print ("Start of evolution")

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

# Evaluate the individuals with an invalid fitness
invalid_ind = [ind for ind in population if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit
    singleFitness.append([fit[0], fit[1], fit[2]])

#Add validation results corelated to min fitness
df = pd.DataFrame(data = singleFitness, columns = ['train', 'val', 'val_acc'])
res = df.loc[df['train'].idxmin()]
wholeFitness.append((0, res.iloc[0], res.iloc[1], res.iloc[2]))

if halloffame is not None:
    halloffame.update(population)

record = stats.compile(population) if stats is not None else {}
logbook.record(gen=0, nevals=len(invalid_ind), **record)
if verbose:
    print (logbook.stream)

# Begin the generational process
for gen in range(1, ngen + 1):
    # Select the next generation individuals
    offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    singleFitness = []
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        singleFitness.append([fit[0], fit[1], fit[2]])

    #Add validation results corelated to min fitness
    df = pd.DataFrame(data = singleFitness, columns = ['train', 'val', 'val_acc'])
    res = df.loc[df['train'].idxmin()]
    wholeFitness.append((gen, res.iloc[0], res.iloc[1], res.iloc[2]))

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(offspring)

    # Replace the current population by the offspring
    population[:] = toolbox.select(population + offspring, mu)

    # Append the current generation statistics to the logbook
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    if verbose:
        print (logbook.stream)
      
    #EarlyStopping according to https://heal.heuristiclab.com/system/files/diss%20gkr2.pdf
    if gen > earlyStoppingGens:
        #srcData = wholeFitness[-earlyStoppingGens::]
        #MA = sum(row[2] for row in srcData)/earlyStoppingGens
        #threshold = (1 - earlyStoppingThresh)*wholeFitness[-earlyStoppingGens:-earlyStoppingGens+1][0][2]
        #if MA < threshold:
        pprint.pprint(wholeFitness[-1])
        pprint.pprint(wholeFitness[-1][0])
        if wholeFitness[-earlyStoppingGens][2] < wholeFitness[-1][2]:
            print ("Early Stopping accoured gen = " + repr(gen))
            pprint.pprint(wholeFitness[-1])
            pprint.pprint(wholeFitness[-earlyStoppingGens:-earlyStoppingGens+1])
            break

print ("-- End of (successful) evolution --")


# In[ ]:


dfWF = pd.DataFrame(data = wholeFitness, columns = ['gen','train', 'val', 'val_acc'])
display(dfWF.loc[dfWF['train'].idxmin()])
display(dfWF.loc[dfWF['val'].idxmin()])
display(dfWF.loc[dfWF['val_acc'].idxmax()])


# In[ ]:


gen = dfWF["gen"]
fit_mins = dfWF["train"]
val_mins = dfWF["val"]
val_acc = dfWF["val_acc"]

fig, ax1 = plt.subplots(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
line1 = ax1.plot(gen, fit_mins, "b-", label="Train Fitness")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Train Fitness", color="b")
for tl in ax1.get_yticklabels():
    tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gen, val_mins, "r-", label="Val Fitness")
ax2.set_ylabel("Val Fitness", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

ax3 = ax1.twinx()
line3 = ax3.plot(gen, val_acc, "y-", label="Val Acc")
ax3.set_ylabel("Val Acc", color="y")
for tl in ax3.get_yticklabels():
    tl.set_color("y")
    
lns = line1 + line2 + line3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc="center right")

plt.show()


# In[ ]:


func = toolbox.compile(expr=halloffame[0])

predictions = []  
for index, point in X_test.iterrows():
    pred = bool(func(*point[:]))
    predictions.append(pred)

accuracy_score(Y_test, predictions)


# In[ ]:


display(str(halloffame[0]))


# In[ ]:


predictions = []

for index, point in X_pred.iterrows():
    pred = bool(func(*point[:]))
    predictions.append(pred)

df_test['Survived'] = predictions
df_test['Survived'] = df_test['Survived'].astype('int')
submit = df_test[['PassengerId','Survived']]
submit.to_csv("../working/submit22.csv", index=False)

