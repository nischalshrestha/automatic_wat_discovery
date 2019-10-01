#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Changelog:
# 27/04/2016 - forked kaggle-titanic-001 notebook from Michiel Kalkman + added some exploratory plots + feature importance (see IV)


### RESOURCES:
# (I) - https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic/
# (II) - https://gist.github.com/mwaskom/8224591
# (II) - https://stanford.edu/~mwaskom/software/seaborn/tutorial/categorical.html
# (IV) - http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html



#0.a - IMPORT libraries and read train and test set:
import numpy as np
import pandas as pd


train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, ) #is a panda df
test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )  #is a panda df
#0.b - HELPER FUNCTION TO HANDLE MISSING DATA 
def harmonize_data(titanic):

    
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()
    
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    
    titanic["Embarked"] = titanic["Embarked"].fillna("S")#fill the two missing values with the most occurred value, which is "S".

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    
    titanic.drop("Cabin",axis=1,inplace=True)

    return titanic
#1.a - CLEANING DATA:
print("------------ ORIGINAL TRAIN DF")
print(train.head(3)) #display some of the DF
train_data = harmonize_data(train) 
test_data  = harmonize_data(test)
print("------------ HARMONIZED DF")
print(train_data.head(3)) #Notice that "Cabin" has been removed
print("------------ ORIGINAL TEST DF")
print(test.head(3)) #display some of the DF


# In[ ]:



#1.b - GETTING FAMILIAR WITH THE DATA:
all_data = pd.concat([test_data,train_data])#see http://pandas.pydata.org/pandas-docs/stable/merging.html 
print('Size of the complete datase: ', all_data.shape)


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

# Survival Rate as function of embark port
d1 = sns.factorplot('Embarked','Survived', data=all_data,size=4,aspect=3)
(d1.set_axis_labels("", "Survival Rate")
   .set_xticklabels(["S", "C", "Q"]))  



# In[ ]:


#Survival rate for men, women as function of class
#hints: http://stanford.edu/~mwaskom/software/seaborn-dev/generated/seaborn.factorplot.html
d2 = sns.factorplot("Sex", "Survived", col="Pclass",data=all_data, saturation=.5,kind="bar", ci=None, aspect=.6)
(d2.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_var} {col_name}")
    .set(ylim=(0, 1))
    .despine(left=True))  


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Age distribution of Survived Passengers')
axis2.set_title('Age distribution of Non-Survived Passengers')

#Age histogram 
plotValues = all_data['Age'].astype(int)
weights = np.ones_like(plotValues)/len(plotValues) #see: http://stackoverflow.com/questions/3866520/plotting-histograms-whose-bar-heights-sum-to-1-in-matplotlib?lq=1
plotValues.hist(bins=10, weights=weights,ax = axis1)

plotValues_notSurvived = all_data[all_data['Survived']==0]['Age']
weights = np.ones_like(plotValues_notSurvived)/len(plotValues_notSurvived)
#2nd Plot - overlay - "bottom" series
plotValues_notSurvived.hist(bins=10,weights=weights, ax = axis2, color='red')

#A first visualization by age
#d3 = sns.violinplot(x="Sex",y="Age", hue="Survived", data=all_data, inner='points',ax=axis2)
#d3.set_xticklabels(["Men", "Women"])





# In[ ]:


#A better visualization by age
d4 = sns.FacetGrid(all_data, hue="Survived",aspect=4)
d4.map(sns.kdeplot,'Age',shade= True) #kernel density estimation
d4.set(xlim=(0, all_data['Age'].max()))
d4.add_legend()


# In[ ]:


# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))

all_data["Age"] = all_data["Age"].astype(int)
average_age = all_data[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
d5 = sns.barplot(x='Age', y='Survived', data=average_age)


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

algForest = RandomForestClassifier( #http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
    random_state=1,
    n_estimators=160,
    min_samples_split=6,
    min_samples_leaf=2
)
algForest.fit(train_data[predictors],train_data["Survived"])
#assessing feature importance
importances = algForest.feature_importances_
std = np.std([importances for algForest in algForest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(len(predictors)):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(predictors)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(predictors)), predictors)
plt.xlim([-1, len(predictors)])
plt.show()
#Feature importance from: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html



# In[ ]:



def create_submission(alg, train, test, predictors, filename):

    alg.fit(train[predictors], train["Survived"])
    predictions = alg.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
    submission.to_csv(filename, index=False)
create_submission(alg, train_data, test_data, predictors, "run-01.csv")


# In[ ]:



def create_submission(alg, train, test, predictors, filename):
    alg.fit(train[predictors], train["Survived"])
    predictions = alg.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
    submission.to_csv(filename, index=False)
create_submission(alg, train_data, test_data, predictors, "run-01.csv")


# In[ ]:




