#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import matplotlib.pyplot as plt 
from sklearn.neural_network import MLPClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("../input/train.csv")
train_df = train.drop("Survived", axis=1) 
test_df = pd.read_csv("../input/test.csv")

#Combine the two datasets for data processing
comb = pd.concat([train_df,test_df])
comb = comb.reset_index()
comb


# In[ ]:


#Extract Title from Name
comb["Title"] = comb.Name.str.extract('^.* ([A-Z][a-z]+)\..*')
#comb.Title.value_counts()
pd.crosstab(comb.Sex, comb.Title)
comb.Title.loc[(comb.Title.isin(['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Dr'])) & (comb.Sex=='male')]  = 'Sir'
comb.Title.loc[(comb.Title.isin(['Countess', 'Dona', 'Lady', 'Mlle', 'Mme', 'Dr'])) & (comb.Sex=='female')] = 'Mme'
comb.Title.loc[comb.Title.isin(['Ms'])] = 'Miss'
comb.Title.value_counts()


# In[ ]:


#Look for NaN Embarked samples
print(comb[comb.Embarked.isnull()])


# In[ ]:


#Look for appropriate Embarked based on Fare(80) and Pclass(1)
data = comb[["Fare","Pclass","Embarked"]]
fig = data.boxplot(by=["Pclass","Embarked"], grid=True)
fig.set_yticks([80])
fig.set_ylim(0,100)


# In[ ]:


comb.Embarked.loc[comb.Embarked.isnull()] = 'C'
comb.info()


# In[ ]:


comb[comb.Fare.isnull()]


# In[ ]:


plt.figure(figsize=(9,2))
mean = comb.Fare[(comb.Pclass==3) & (comb.Embarked=='S')].mean()
med = comb.Fare[(comb.Pclass==3) & (comb.Embarked=='S')].median()
data=comb["Fare"][(comb.Pclass==3) & (comb.Embarked=='S')]
data.plot(kind="kde")
plt.axvline(med, color='red')
plt.axvline(mean, color='green')


# In[ ]:


#Set Fare to median of corresponding Pclass and Embarked
comb.Fare[comb.Fare.isnull()] = comb.Fare[(comb.Pclass==3) & (comb.Embarked=='S')].median()
comb.info()


# In[ ]:


#Create FamilySize from Parch and SibSp + 1 (incl. themselves)
comb["FamilySize"] = comb.Parch + comb.SibSp + 1

#Change categorical variables to factors
gender_dummies = pd.get_dummies(comb.Sex, drop_first=True)
embarked_dummies = pd.get_dummies(comb.Embarked)
comb = comb.join(gender_dummies)
comb = comb.join(embarked_dummies)
comb = comb.drop(["Pclass", "Sex", "Embarked"], axis=1)
comb.info()


# In[ ]:


#Fill in Age NaN values by regressing Age ~ other predictors
age_model = DecisionTreeRegressor(min_samples_split=5, random_state=42)
test = comb
comb = comb.join(pd.get_dummies(comb.Title))
comb = comb.drop(["Title", "Name", "PassengerId", "Ticket", "Cabin", "SibSp", "Parch"], axis=1)
x_train = comb.drop("Age",axis=1)[comb.Age.notnull()]
x_test = comb.drop("Age",axis=1)[comb.Age.isnull()]
y_train = comb["Age"][comb.Age.notnull()]
age_model.fit(x_train,y_train)
print(age_model.score(x_train, y_train))
y_test = age_model.predict(x_test)

#Convert np.array to Series
y_test_series = pd.Series(y_test)

#Plot new ages compared to old ages to find any changes in the shape of the age distribution
age_reg = y_train.append(y_test_series)
age_reg.plot(kind="hist", alpha=0.55, bins=70, legend=True, label='Predicted')
y_train.plot(kind="hist", alpha=0.55, bins=70, legend=True, label='Known')


# In[ ]:


comb.Age[comb.Age.isnull()]=y_test
comb.info()


# In[ ]:


x = comb.loc[:890].reset_index(drop=True)
x_test = comb.loc[891:].reset_index(drop=True)
x = x.drop("index", axis=1)
x_test = x_test.drop("index", axis=1)
x.info()


# In[ ]:


y = train.Survived
x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.25, random_state=0)


# In[ ]:


model = DecisionTreeClassifier(min_samples_split = 10)
model.fit(x_train,y_train)
print("Decision Tree")
print(model.score(x_train,y_train))
print(model.score(x_cv,y_cv))

imp = pd.DataFrame({"Features":x_train.columns})
imp["DecTree"] = model.feature_importances_

print("----------------------------")

rf = RandomForestClassifier(min_samples_split =20, n_estimators=100)
rf.fit(x_train,y_train)
print("Random Forest")
print(rf.score(x_train,y_train))
print(rf.score(x_cv,y_cv))
imp["RandForest"] = rf.feature_importances_
print(imp)


# In[ ]:


y_test = rf.predict(x_test)
submission = pd.DataFrame({"PassengerId":test_df.PassengerId, "Survived":y_test})
submission.to_csv("titanic_submission.csv", index=False)


# In[ ]:


fig = imp.plot(kind="barh", alpha = 0.50)
fig.set_yticklabels(imp.Features)


# In[ ]:


#Plot a learning curve to find if the model is overfitting or underfitting
#Because the two curves are so close when m os large, the model has high bias (underfitting)
#A solution to this is to add more features to increase complexity

train_sizes, train_scores, cv_scores = learning_curve(rf, x, y, train_sizes=range(1,751,50), cv=5)
#data = pd.DataFrame({"train_sizes":train_sizes, "train_scores":train_scores, "cv_scores":cv_scores})
train_scores = 1-train_scores.mean(axis=1)
cv_scores = 1-cv_scores.mean(axis=1)
data = pd.DataFrame({"train_sizes":train_sizes})
data["train_scores"] = train_scores
data["cv_scores"] = cv_scores
fig = data.plot(x="train_sizes", kind="line")
fig.grid(True)

