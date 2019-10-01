#!/usr/bin/env python
# coding: utf-8

# # Description
# Analyzing data, selecting most correlating features and applying decision tree  to predict survival.

# ## predefs

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt
import numpy.random as rnd
from sklearn import tree as decision_tree
import graphviz

def prepare_dataset(input_data):
    input_data["Sex"] = input_data["Sex"].apply(lambda x: int(x == "male"))
    input_data["pclass_1"] = input_data["Pclass"].apply(lambda x: int(x == 1))
    input_data["pclass_2"] = input_data["Pclass"].apply(lambda x: int(x == 2))
    input_data["pclass_3"] = input_data["Pclass"].apply(lambda x: int(x == 3))
    input_data["Embarked_C"] = input_data["Embarked"].apply(lambda x: int(x == "C"))
    input_data["Embarked_Q"] = input_data["Embarked"].apply(lambda x: int(x == "Q"))
    input_data["Embarked_S"] = input_data["Embarked"].apply(lambda x: int(x == "S"))
    return input_data


# ## Import Data

# In[ ]:


train_data = prepare_dataset(pd.read_csv("../input/train.csv"))

print(train_data.head(3))
train_data.hist(column="Survived")


# ## finding correlating features

# In[ ]:


feature_candidates = [
    "pclass_1", "pclass_2", "pclass_3", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"
]

# noise data to see the distribution better
plot_data = train_data.copy()
for feature in ["Survived"] + feature_candidates:
    plot_data[feature] = plot_data[feature].apply(lambda x: x + rnd.rand() * 0.5)

# render feature-wise scatter plots
for feature in feature_candidates:
    plot_data.plot.scatter(x=feature, y="Survived")
    
for feature in feature_candidates:
    print(feature, "missing:", sum(train_data[feature].isnull()))


# In[ ]:


# selecting good features
features = [
    "pclass_3", "Sex", "Fare"
]


# ## training classifier

# In[ ]:


reduced = train_data[["Survived"] + features].dropna()
classifier = decision_tree.DecisionTreeClassifier(min_samples_leaf=10)
classifier = classifier.fit(reduced[features], reduced[["Survived"]])
graphviz.Source(decision_tree.export_graphviz(classifier, out_file=None)).render("tree")
print("train data got:", len(train_data))
print("data trained:", len(reduced))


# ## testing classifier

# In[ ]:


test_data = prepare_dataset(pd.read_csv("../input/test.csv")).fillna(0)

predictions = classifier.predict(test_data[features])
test_data["Survived"] = predictions
print("survived:", sum(x == 1 for x in predictions))
print("not survived:", sum(x == 0 for x in predictions))
test_data[["PassengerId", "Survived"]].to_csv("data.csv", index=False)
    


# In[ ]:




