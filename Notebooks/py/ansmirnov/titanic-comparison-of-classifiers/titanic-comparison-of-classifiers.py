#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
mark="Survived"


# In[ ]:


def harmonize_data(titanic):
    
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()
    
    
    
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    
    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic


# In[ ]:


def create_submission(clf, train, test, predictors, filename):

    clf.fit(train[predictors], train["Survived"])
    predictions = clf.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
    
    submission.to_csv(filename, index=False)


# In[ ]:


from sklearn.model_selection import cross_val_score

def validation_scores(clf, train_data):
    scores = cross_val_score(
        clf,
        train_data[predictors],
        train_data[mark],
        cv=3
    )
    return scores.mean()


# In[ ]:


train_data = harmonize_data(train)
test_data  = harmonize_data(test)


# In[ ]:


def compare_metods(classifiers, train_data):
    names, scores = [], []
    for name, clf in classifiers:
        #names.append(str(alg))
        names.append(name)
        scores.append(validation_scores(clf, train_data))
    return pd.DataFrame(scores, index=names, columns=['Scores'])


# In[ ]:


# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    
classifiers = [
    ("Nearest Neighbors", KNeighborsClassifier(3)),
    ("Linear SVM", SVC(kernel="linear", C=0.025)),
    ("RBF SVM", SVC(gamma=2, C=1)),
    ("Gaussian Process",GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    ("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("Neural Net", MLPClassifier(alpha=1)),
    ("AdaBoost",AdaBoostClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("QDA", QuadraticDiscriminantAnalysis())
]

res = compare_metods(classifiers, train_data)
res


# In[ ]:


get_ipython().magic(u'matplotlib inline')

#import matplotlib.pyplot as plt

res.plot(kind='bar', rot=90)


# In[ ]:


best_alg = classifiers[sorted(list(zip(list(res.Scores), range(len(res)))))[-1][1]]
create_submission(best_alg[1], train_data, test_data, predictors, "titanic.csv")

