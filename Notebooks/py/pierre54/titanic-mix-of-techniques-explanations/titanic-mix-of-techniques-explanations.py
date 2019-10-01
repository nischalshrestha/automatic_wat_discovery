#!/usr/bin/env python
# coding: utf-8

# Hi everyone!<br>
# I posted some minutes ago my first script on Kaggle and, now, I open a notebook to comment the choices I made.<br>
# I'm a beginner in data science (and in English) so all comments are welcome!

# # Discovering of the dataset

# In[ ]:


import pandas as pd
train = pd.read_csv("../input/train.csv")
train.info()


# In[ ]:


train.describe()


# In[ ]:


train.head()


# The two functions info and describe give us some interesting informations about the dataset and I used the head function to have an idea of them.<br>
# I made the choice to categorize them into 6 categories : Pid (passenger id), Class (it is not the class of the passenger but correspond to the Survived column), Continuous, Text, Discrete, Dummy (discrete variables with only two options).

# In[ ]:


Pid = "PassengerId"
Class = "Survived"
Continuous = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
Text = ["Name", "Ticket", "Cabin"]
Discrete = "Embarked"
Dummy = "Sex"


# # Field empty cells
# The info function indicates that the columns Age, Cabin and Embarked contain missing value. I saw there is some methods to take care of the missing value : delete the column, delete the row or add an approximate value.<br>
# In the case of the "Cabin" column, we only have 204 values out of 891. Furthermore, it contains text who is difficult to handle so I chose to delete the column.
# In the case of the "Embarked" column, there are only two missing values. There is a temptation to delete the concerned rows but with only 891 values, I prefer to fill it by the most present value in the dateset.
# In the case of the "Age" column, we have 714 values out of 891. For the same reason, we will not delete the rows. But, this time, we have an important and easy-to-use information. So I chose to keep the column and fill empty cell by the mean value of the dataset.<br>
# I thought to make a linear regression to compute an approximate value for the Age column but I never see people did it on the forums I read; what do you think of this idea?

# In[ ]:


train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].value_counts().idxmax())

# ---- From now, I will use data in categorical variables ----- #
train_pid = train[Pid]
train_class = train[Class]
train_continuous = train[Continuous]
train_text = train[Text]
train_discrete = train[Discrete]
train_dummy = train[Dummy]


# # Adding new data
# In this section, there is no surprise of you read other notebooks: I add two columns, one based on the title in the Name column, another based on the size of the family (SibSp + Parch + 1).<br>
# I also chose to categorize people in function of their title and only create two categories: regular title and the others in order to limit the number of variables in my model and avoid bad generalization.
# In many notebooks, I saw people using family name to create a new column. I didn't do it because I don't think we can talk about learning in this case. It will work in this case because the test set contains people from the same boat that the train set. But, now, let's imagine we want to create a model who can predict the probability of a person to die during a cruise. This time, the family name is totally useless. Your idea about this subject?<br>

# In[ ]:


import re

def getTitle(x):
    result = re.compile(r'.*?,(.*?)\.').search(x)
    if result:
        return result.group(1).strip()
    else:
        return ''

train_family_size = train["SibSp"] + train["Parch"] + 1
train_continuous = pd.concat([train_continuous, train_family_size], axis=1)

train_title = train["Name"].apply(getTitle)
title_mapping = {"Mr": 0, "Miss": 0, "Mrs": 0, "Master": 0, "Dr": 1, "Rev": 1, "Major": 1, "Col": 1, "Mlle": 0, "Mme": 0, "Don": 1, "Lady": 1, "the Countess": 1, "Jonkheer": 1, "Sir": 1, "Capt": 1, "Ms": 0, "Dona": 1, "": 0}

for k,v in title_mapping.items():
    train_title[train_title == k] = v

train_continuous = pd.concat([train_continuous, train_title], axis=1)


# # Normalization of the data
# Because I want to use different algorithms to create my model, my continuous data have to be normalized.

# In[ ]:


from sklearn import preprocessing as pp
minmax_scaler = pp.MinMaxScaler((0,1)).fit(train_continuous)
train_continuous = pd.DataFrame(minmax_scaler.transform(train_continuous))


# # Transformation of the other variables
# The dummy variables as the Sex column will be replaced by 0 and 1 value and the discrete variables as the Embarked columns will be replace same way but we will create many columns there are different values (in this case, we will create 3 columns). That's why we didn't create a lot of categories for the title column.

# In[ ]:


train_discrete = pd.get_dummies(train_discrete)

lb = pp.LabelBinarizer()
lb.fit(train_dummy)
train_dummy = pd.DataFrame(lb.transform(train_dummy))

# ---- For information ... ----- #
train_discrete.head()


# Now, we can merge all the arrays to create the input of our model. I don't use the Ticket column because I found to hard to handle and I don't really see what it can bring to our model.

# In[ ]:


X = pd.concat([train_continuous, train_discrete, train_dummy], axis=1)


# # Creation of the model
# I don't really now how to choose the best algorithm in function of the situation. In this case, I preferred to use aggregation in order to combine the forces (and the weakness) of all of them. Here is the list of them and the explanation of my choice:<br>
#  - Random Forest: according me, it's the best method to find the perfect model but I'm scared by its trend to overfit;<br>
#  - Nearest Neighbors: I was thinking of the situation, you are on a falling down boat and you want to survive, what do you do? You follow people who are going on safety boat and closer you are from a survival, higher are your chance to survive. It's an image but I guess this algorithm can be a good idea in this case. I used cross validation to choose the number of nearest neighbors K;<br>
#  - Multi-layer Perceptron: I hope this algorithm will see weak signals in the data;<br>
#  - Support vector machine (with kernel method): the perfect algorithm to find non-linear model but I used cross-validation to avoid overfit;<br>
# - Logistic regression: I wanted a fifth algorithm so I chose this one with no particular reason.<br>
# I would like to know what do you think of my choices. Good idea? Bad idea? Idea of improvement?

# In[ ]:


import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model as lm

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, train_class)

knc = KNeighborsClassifier()
Cs = np.linspace(1, 19, 10).astype(int)
neigh = GridSearchCV(estimator=knc, param_grid=dict(n_neighbors=Cs), cv=10, n_jobs=-1)
neigh.fit(X, train_class)

mlp = MLPClassifier()
mlp.fit(X, train_class)

svc = svm.SVC()
Cs = np.logspace(-6, 2)
svc = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv=10, n_jobs=-1)
svc.fit(X, train_class)

lr = lm.LogisticRegression()
lr.fit(X, train_class)


# # Predictions
# Now, we will do the same work on the test set. I think it's more common to used data from the train set to fill empty cell in the test set. What do you think about that?

# In[ ]:


test = pd.read_csv("../input/test.csv")

test["Age"] = test["Age"].fillna(train["Age"].mean())
test["Fare"] = test["Fare"].fillna(train["Fare"].mean())

test_pid = test[Pid]
test_continuous = test[Continuous]
test_text = test[Text]
test_discrete = test[Discrete]
test_dummy = test[Dummy]

test_family_size = test["SibSp"] + test["Parch"] + 1
test_continuous = pd.concat([test_continuous, test_family_size], axis=1)

test_title = test["Name"].apply(getTitle)

for k,v in title_mapping.items():
    test_title[test_title == k] = v

test_continuous = pd.concat([test_continuous, test_title], axis=1)

test_continuous = pd.DataFrame(minmax_scaler.transform(test_continuous))

test_discrete = pd.get_dummies(test_discrete)

test_dummy = pd.DataFrame(lb.transform(test_dummy))

X = pd.concat([test_pid, test_continuous, test_discrete, test_dummy], axis=1)


# And to finish, I create the CSV file.<br>
# I didn't use preexisting class for the aggregation so I make the sum of all the results and put 1 if greater or equal 3, 0 else.

# In[ ]:


import csv as csv
result_file = open("result.csv", "w")
result_file_obj = csv.writer(result_file)
result_file_obj.writerow(["PassengerId", "Survived"])
for idx, row in X.iterrows():
	if(rfc.predict(row[1::].reshape(1, -1))[0] + neigh.predict(row[1::].reshape(1, -1))[0] + mlp.predict(row[1::].reshape(1, -1))[0] + svc.predict(row[1::].reshape(1, -1))[0] + lr.predict(row[1::].reshape(1, -1))[0] >= 3):
		result_file_obj.writerow([row["PassengerId"].astype(int), 1])
	else:
		result_file_obj.writerow([row["PassengerId"].astype(int), 0])

result_file.close()

