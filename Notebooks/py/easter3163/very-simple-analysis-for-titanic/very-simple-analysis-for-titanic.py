#!/usr/bin/env python
# coding: utf-8

# # Titanic : Machine Learning form Disaster

# ## Overview
# * training set(train.csv)
# * test set(test.csv)

# ## Data Dictionary
# |  <center>Variable</center> |  <center>Definition</center> |  <center>key</center> |
# |:--------|:--------:|--------:|
# |**survival** | <center>Survival </center> |- |
# |**pclass** | <center>Ticket class </center> |- |
# |**sex** | <center>Sex </center> |- |
# |**Age** | <center>Age in years </center> |-|
# |**sibsp** | <center># of siblings / spouses aboard the Titanic </center> |- |
# |**parch** | <center># of parents / children aboard the Titanic </center> |- |
# |**ticket** | <center>Ticket number </center> |- |
# |**fare** | <center>Passenger fare </center> |-|
# |**cabin** | <center>Cabin number </center> |- |
# |**embarked** | <center>Port of Embarkation </center> |C = Cherbourg, Q = Queenstown, S = Southampton |

# ## Variable Notes
# * pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# * age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# ## Load Dataset
# * You can use read_csv in Pandas

# In[ ]:


import pandas as pd
train = pd.read_csv("../input/train.csv", index_col = "PassengerId")
test = pd.read_csv("../input/test.csv", index_col="PassengerId")


# In[ ]:


# Check the row and column of train dataframe
train.shape


# In[ ]:


train.head()


# In[ ]:


test.shape


# In[ ]:


test.head()


# ## Explore
# * Let 's visualize our dataset(train, test) by using matplotlib and seaborn

# In[ ]:


get_ipython().magic(u'matplotlib inline')

import seaborn as sns
import matplotlib.pyplot as plt


# ### Sex
# * We can use countplot in this column

# In[ ]:


sns.countplot(data=train, x="Sex", hue='Survived')


# * **female > male** is True

# In[ ]:


pd.pivot_table(train, index="Sex", values="Survived")


# ### Pclass
# * We use countplot again in this column

# In[ ]:


sns.countplot(data=train, x="Pclass", hue="Survived")


# * **If Pclass high, probability of survival is high.** 

# In[ ]:


pd.pivot_table(train, index="Pclass", values="Survived")


# ### Embarked
# * There are three types of marina: Cherbourg (C) 2) Queenstown (Q) 3) Southampton (S).

# In[ ]:


sns.countplot(data=train, x="Embarked", hue="Survived")


# * The more you board in Cherbourg (C), the more likely you are to survive, and the more likely you are to board in Southampton (S), the more likely you are to die

# In[ ]:


pd.pivot_table(train, index="Embarked", values="Survived")


# ### Age & Fare
# * Let 's use seaborn's lmplot

# In[ ]:


# If you don't want Regression, make fit_reg False
sns.lmplot(data=train, x="Age", y="Fare", hue="Survived", fit_reg=False)


# In[ ]:


low_low_fare = train[train["Fare"] < 100]


# In[ ]:


sns.lmplot(data=low_low_fare, x="Age", y="Fare", hue="Survived", fit_reg=False)


# * Looking closely at the results, passengers ages 15 and younger are more likely to survive, and passengers paying less than $ 20 for freight rates have a significantly higher chance of survival.

# ### SipSp, Parch
# * So add SibSp and Parch to see the total number of family members (FamilySize).

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
train[["FamilySize"]].head()


# In[ ]:


sns.countplot(data=train, x="FamilySize", hue="Survived")


# 
# * If you board the Titanic alone (FamilySize == 1), your chances of survival are very low.
# * If the Titanic boarded a family of suitable people (2 <= FamilySize <= 4), the probability of survival is relatively high.
# * However, if the number of family members on the Titanic is too large (FamilySize> = 5), you will find that the chances of survival are very low.

# In[ ]:


train.loc[train["FamilySize"]==1, "FamilyType"] = "Single"
train.loc[(train["FamilySize"] > 1) & (train["FamilySize"] < 5), "FamilyType"] = "Nuclear"
train.loc[train["FamilySize"] >=5, "FamilyType"] = "Big"
train[["FamilySize", "FamilyType"]].head()


# In[ ]:


sns.countplot(data=train, x="FamilyType", hue="Survived")


# * The analysis shows that the survival rate of the nuclear family is high and the survival rate of the other two types (Single, Big) is significantly lower.

# In[ ]:


pd.pivot_table(data=train, index="FamilyType", values="Survived")


# * The results show that the survival rate is only 30.3% for Single, 57.8% for Nuclear, and 16.1% for Big.

# ### Name
# 
# * , The part before is the last name (SurName)
# * , And. The part in between is the passenger's title.
# * Finally . The part after the name is FirstName.

# In[ ]:


train["Name"].head()


# In[ ]:


def get_title(name):
    return name.split(", ")[1].split(". ")[0]
train["Name"].apply(get_title).unique()


# In[ ]:


train.loc[train["Name"].str.contains("Mr"), "Title"] = "Mr"
train.loc[train["Name"].str.contains("Miss"), "Title"] = "Miss"
train.loc[train["Name"].str.contains("Mrs"), "Title"] = "Mrs"
train.loc[train["Name"].str.contains("Master"), "Title"] = "Master"

train[["Name", "Title"]].head()


# In[ ]:


sns.countplot(data=train, x="Title", hue="Survived")


# In[ ]:


pd.pivot_table(train, index="Title", values="Survived")


# ## Preprocessing
# ### Encode Sex

# In[ ]:


train.loc[train["Sex"] == "male", "Sex_encode"] = 0
train.loc[train["Sex"] == "female", "Sex_encode"] = 1

train[["Sex", "Sex_encode"]].head()


# * preprocess test data also

# In[ ]:


test.loc[test["Sex"] == "male", "Sex_encode"] = 0
test.loc[test["Sex"] == "female", "Sex_encode"] = 1

test[["Sex", "Sex_encode"]].head()


# ### Filling Data(fare)

# In[ ]:


train[train["Fare"].isnull()]


# In[ ]:


test[test["Fare"].isnull()]


# * Since only one value is empty in the entire test data, it seems that even if you insert the appropriate value instead of the average, it works well without any loss of accuracy.

# In[ ]:


train["Fare_fillin"] = train["Fare"]
test["Fare_fillin"] = test["Fare"]


# In[ ]:


test.loc[test["Fare"].isnull(), "Fare_fillin"] = 0


# In[ ]:


train["Fare_fillin"] = train["Fare_fillin"] / 10.0
test["Fare_fillin"] = test["Fare_fillin"] / 10.0


# ### Encode Embarked

# * C == [True, False, False]
# * S == [False, True, False]
# * Q == [False, False, True]

# In[ ]:


train["Embarked"].fillna("S")


# In[ ]:


train["Embarked_C"] = False
train.loc[train["Embarked"]=='C', "Embarked_C"] = True
train["Embarked_S"] = False
train.loc[train["Embarked"]=='S', "Embarked_S"] = True
train["Embarked_Q"] = False
train.loc[train["Embarked"]=='Q', "Embarked_Q"] = True
train[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()


# In[ ]:


test["Embarked_C"] = False
test.loc[test["Embarked"]=='C', "Embarked_C"] = True
test["Embarked_S"] = False
test.loc[test["Embarked"]=='S', "Embarked_S"] = True
test["Embarked_Q"] = False
test.loc[test["Embarked"]=='Q', "Embarked_Q"] = True
test[["Embarked", "Embarked_C", "Embarked_S", "Embarked_Q"]].head()


# ### Encode Age

# In[ ]:


train["Age"].fillna(train["Age"].mean(), inplace=True)
test["Age"].fillna(test["Age"].mean(), inplace=True)


# In[ ]:


train["Child"] = False
train.loc[train["Age"] < 15, "Child"] = True
train[["Age", "Child"]].head(10)


# In[ ]:


test["Child"] = False
test.loc[test["Age"] < 15, "Child"] = True
test[["Age", "Child"]].head(10)


# ### Encode FamilySize

# In[ ]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[ ]:


test[["FamilySize"]].head()


# In[ ]:


train["Single"] = False
train.loc[train["FamilySize"]==1, "Single"] = True
train["Nuclear"] = False
train.loc[(train["FamilySize"]>1)&(train["FamilySize"]<5), "Nuclear"] = True
train["Big"] = False
train.loc[train["FamilySize"] >=5, "Big"] = True
train[["FamilySize", "Single", "Nuclear", "Big"]].head(10)


# In[ ]:


test["Single"] = False
test.loc[test["FamilySize"]==1, "Single"] = True
test["Nuclear"] = False
test.loc[(test["FamilySize"]>1)&(test["FamilySize"]<5), "Nuclear"] = True
test["Big"] = False
test.loc[test["FamilySize"] >=5, "Big"] = True
test[["FamilySize", "Single", "Nuclear", "Big"]].head(10)


# ### Encode Name

# In[ ]:


train["Master"] = False
train.loc[train["Name"].str.contains("Master"), "Master"] = True
train[["Name", "Master"]].head(10)


# In[ ]:


test["Master"] = False
test.loc[test["Name"].str.contains("Master"), "Master"] = True
test[["Name", "Master"]].head(10)


# ## Train

# In[ ]:


feature_names = ["Pclass", "Sex_encode", "Fare_fillin", "Embarked_C", "Embarked_S", "Embarked_Q", "Child", "Single", "Nuclear", "Big", "Master"]
feature_names


# In[ ]:


label_name = "Survived"
label_name


# In[ ]:


from sklearn.model_selection import train_test_split
random_seed=0
Y = train[label_name]
X = train[feature_names]
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=random_seed)


# In[ ]:


#X_train = X_train[feature_names]
#y_train = y_train[label_name]
X_test = test[feature_names]


# In[ ]:


# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV


# ### Random Forest Model

# In[ ]:


rforest_model = RandomForestClassifier(n_estimators=100)


# ### Support Vector Machine Model

# In[ ]:


svc = SVC()


# ### K nearest neighbors Model

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)


# ### Gradient Boosting Classifier

# In[ ]:


gbmodel = GradientBoostingClassifier(max_depth=12)


# ### Gaussian Naive Bayes

# In[ ]:


gnb = GaussianNB()


# ### Logistic Regression

# In[ ]:


LR = LogisticRegression()


# ### MLP Classifier

# In[ ]:


mlp = MLPClassifier(solver='lbfgs', random_state=0)


# ### NN

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# In[ ]:


x_train = X_train.values
x_val = X_val.values
test_nn = test[feature_names].values


# In[ ]:


model = Sequential()

batch_size = 32
epochs = 200

model.add(Dense(32, activation="relu", input_dim=11))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])


# In[ ]:


model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size)


# In[ ]:


test_loss, test_acc = model.evaluate(x_val, y_val)
print('Test Score:{}'.format(test_acc))


# In[ ]:


nn_predict = model.predict(test_nn)


# In[ ]:


rforest_model.fit(X_train, y_train)
svc.fit(X_train, y_train)
knn.fit(X_train, y_train)
gbmodel.fit(X_train, y_train)
gnb.fit(X_train, y_train)
LR.fit(X_train, y_train)
mlp.fit(X_train, y_train)


# In[ ]:


score = []
score.append(rforest_model.score(X_val, y_val))
score.append(svc.score(X_val, y_val))
score.append(knn.score(X_val, y_val))
score.append(gbmodel.score(X_val, y_val))
score.append(gnb.score(X_val, y_val))
score.append(LR.score(X_val, y_val))
score.append(mlp.score(X_val,y_val))


# In[ ]:


score


# In[ ]:


score = []
score.append(rforest_model.score(X_train, y_train))
score.append(svc.score(X_train, y_train))
score.append(knn.score(X_train, y_train))
score.append(gbmodel.score(X_train, y_train))
score.append(gnb.score(X_train, y_train))
score.append(LR.score(X_train, y_train))
score.append(mlp.score(X_train,y_train))


# In[ ]:


score


# * I will choose Gradeint boosting model

# In[ ]:


x = X.values
y = Y.values


# In[ ]:


model.fit(x,y, epochs=epochs, batch_size=batch_size)


# In[ ]:


gbmodel.fit(X, Y)


# In[ ]:


predictions = gbmodel.predict(X_test)


# In[ ]:


#nn_predict = model.predict(test_nn)


# In[ ]:


#predictions = nn_predict


# In[ ]:


#predictions = [0 if pred<0.5 else 1 for pred in predictions ]


# In[ ]:


submission = pd.read_csv('../input/gender_submission.csv')
submission['Survived'] = predictions


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('./simpletitanic.csv', index=False)

