#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The score of this Notebook is 0.79425
import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test  = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )


# In[ ]:


train.head(10)


# In[ ]:


train_corr = train.corr()
train_corr


# In[ ]:


#−1.0 to −0.7 Strong negative correlation
#−0.7 to −0.4 Negative correlation
#−0.4 to −0.2 Weak negative correlation
#−0.2 to +0.2 There is no correlation
#+0.2 to +0.4 Weak positive correlation
#+0.4 to +0.7 Positive correlation
#+0.7 to +1.0 Strong Positive correlation


# In[ ]:


def correct_data(train_data, test_data):
    
    # Make missing values ​​for training data from test data as well
    train_data.Age = train_data.Age.fillna(test_data.Age.median())
    train_data.Fare = train_data.Fare.fillna(test_data.Fare.median())
    
    test_data.Age = test_data.Age.fillna(test_data.Age.median())
    test_data.Fare = test_data.Fare.fillna(test_data.Fare.median())    
    
    train_data = correct_data_common(train_data)
    test_data = correct_data_common(test_data)    

    return train_data,  test_data

def correct_data_common(titanic_data):
    titanic_data.Sex = titanic_data.Sex.replace(['male', 'female'], [0, 1])
    titanic_data.Embarked = titanic_data.Embarked.fillna("S")
    titanic_data.Embarked = titanic_data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
    
    return titanic_data


# In[ ]:


train_data,  test_data = correct_data(train, test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

models = []

models.append(("LogisticRegression",LogisticRegression()))
models.append(("SVC",SVC()))
models.append(("LinearSVC",LinearSVC()))
models.append(("KNeighbors",KNeighborsClassifier()))
models.append(("DecisionTree",DecisionTreeClassifier()))
models.append(("RandomForest",RandomForestClassifier()))
rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                max_depth=10, random_state=0, max_features=None)
models.append(("RandomForest2",rf2))
models.append(("MLPClassifier",MLPClassifier(solver='lbfgs', random_state=0)))


results = []
names = []
for name,model in models:
    result = cross_val_score(model, train_data[predictors], train_data["Survived"],  cv=3)
    names.append(name)
    results.append(result)

for i in range(len(names)):
    print(names[i],results[i].mean())


# In[ ]:


alg = rf2

alg.fit(train_data[predictors], train_data["Survived"])

predictions = alg.predict(test_data[predictors])

submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

submission.to_csv('submission.csv', index=False)

