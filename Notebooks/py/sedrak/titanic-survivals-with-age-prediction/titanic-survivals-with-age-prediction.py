#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import keras
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, Embedding
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# # Age sorts into logical categories

# In[ ]:


train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 14, 25, 35, 60, np.inf]
labels = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# # Map each Age value to a numerical value

# In[ ]:


age_mapping = {'Unknown': None,'Child': 1, 'Teenager': 2, 'Young Adult': 3, 'Adult': 4, 'Senior': 5}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()


# # Dropping the Age feature for now, might change

# In[ ]:


train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# # Map each Sex value to a numerical value

# In[ ]:


sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# In[ ]:


train = train.fillna({"Embarked": "S"})


# # Map each Embarked value to a numerical value

# In[ ]:


embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# # Create a combined group of train and test datasets

# In[ ]:


combine = [train, test]


# # Extract a title for each Name in the train and test datasets

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# # Replace various titles with more common names

# In[ ]:


for dataset in combine:
   
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms','Countess','Miss','Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master','Lady'], 'Mr')


# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# # Map each of the title groups to a numerical value

# In[ ]:


title_mapping = {"Dr": 1, "Mr": 2, "Mrs": 3}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# # Drop the name feature since it contains no more useful information.

# In[ ]:


train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# # Fill in missing Fare value in test set based on mean fare for that Pclass 

# In[ ]:


for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)


# # Map Fare values into groups of numerical values

# In[ ]:


train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])


# # Drop Fare values

# In[ ]:


train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# # Drop Cabin values

# In[ ]:


train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# # Drop Ticket values

# In[ ]:


train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# # Make train data from train with age information

# In[ ]:


train_age = train
modifiedFlights = train_age.dropna()


# # Make test data from train without age information

# In[ ]:


null_columns=train.columns[train.isnull().any()]


# In[ ]:


x_train_age = modifiedFlights.drop(['AgeGroup'], axis = 1)


# In[ ]:


y_train_age = modifiedFlights["AgeGroup"]


# In[ ]:


x_test_AgeGroup = train[train.isnull().any(axis=1)]


# In[ ]:


x_test_age = x_test_AgeGroup.drop(['AgeGroup'], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split

predictors = x_train_age.drop(['PassengerId'], axis=1)
target = y_train_age
x_trainage, x_valage, y_trainage, y_valage = train_test_split(predictors, target, test_size = 0.1, random_state = 0)


# # Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_trainage, y_trainage)
y_predage = gbk.predict(x_valage)
acc_gbkage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gbkage)


# # Stochastic Gradient Descent

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_trainage, y_trainage)
y_predage = sgd.predict(x_valage)
acc_sgdage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_sgdage)


# # KNN or k-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_trainage, y_trainage)
y_predage = knn.predict(x_valage)
acc_knnage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_knnage)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_trainage, y_trainage)
y_predage = randomforest.predict(x_valage)
acc_randomforestage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_randomforestage)


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_trainage, y_trainage)
y_predage = decisiontree.predict(x_valage)
acc_decisiontreeage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_decisiontreeage)


# # Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_trainage, y_trainage)
y_predage = perceptron.predict(x_valage)
acc_perceptronage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_perceptronage)


# # Linear SVC

# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_trainage, y_trainage)
y_predage = linear_svc.predict(x_valage)
acc_linear_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_linear_svcage)


# # Support Vector Machines

# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_trainage, y_trainage)
y_predage = svc.predict(x_valage)
acc_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_svcage)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_trainage, y_trainage)
y_predage = logreg.predict(x_valage)
acc_logregage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_logregage)


# # Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_trainage, y_trainage)
y_predage = gaussian.predict(x_valage)
acc_gaussianage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gaussianage)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbkage, acc_sgdage, acc_knnage, acc_randomforestage, 
              acc_decisiontreeage, acc_perceptronage, acc_linear_svcage, acc_svcage, acc_logregage, 
               acc_gaussianage]})
models.sort_values(by='Score', ascending=False)


# # Age prediction

# In[ ]:


predictions = randomforest.predict(x_test_age.drop('PassengerId', axis=1))


# In[ ]:


k=0
for i in range(891):
    if np.isnan(train_age['AgeGroup'][i]) == True:
        train_age['AgeGroup'][i] = predictions[k]
        k+=1


# # Similar approach for test set

# In[ ]:


test_test_age = test 


# In[ ]:


modifiedFlights = test_test_age.dropna()
null_columns=test.columns[test.isnull().any()]


# In[ ]:


x_test_test_age = modifiedFlights.drop(['AgeGroup'], axis = 1)
y_test_test_age = modifiedFlights["AgeGroup"]

x_test_test_AgeGroup = test[test.isnull().any(axis=1)]
x_tst_test_age = x_test_test_AgeGroup.drop(['AgeGroup'], axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split

predictors = x_test_test_age.drop(['PassengerId'], axis=1)
target = y_test_test_age
x_testage_1, x_valage_1, y_testage_1, y_valage_1 = train_test_split(predictors, target, test_size = 0.1, random_state = 0)


# # Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_testage_1, y_testage_1)
y_predage_1 = gbk.predict(x_valage_1)
acc_gbkage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_gbkage_1)


# # Stochastic Gradient Descent

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_testage_1, y_testage_1)
y_predage_1 = sgd.predict(x_valage_1)
acc_sgdage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_sgdage_1)


# # KNN or k-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_testage_1, y_testage_1)
y_predage_1 = knn.predict(x_valage_1)
acc_knnage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_knnage_1)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_testage_1, y_testage_1)
y_predage_1 = randomforest.predict(x_valage_1)
acc_randomforestage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_randomforestage_1)


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_testage_1, y_testage_1)
y_predage_1 = decisiontree.predict(x_valage_1)
acc_decisiontreeage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_decisiontreeage_1)


# # Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_testage_1, y_testage_1)
y_predage_1 = perceptron.predict(x_valage_1)
acc_perceptronage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_perceptronage_1)


# # Linear SVC

# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_testage_1, y_testage_1)
y_predage_1 = linear_svc.predict(x_valage_1)
acc_linear_svcage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_linear_svcage_1)


# # Support Vector Machines

# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_testage_1, y_testage_1)
y_predage_1 = svc.predict(x_valage_1)
acc_svcage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_svcage_1)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_testage_1, y_testage_1)
y_predage_1 = logreg.predict(x_valage_1)
acc_logregage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_logregage_1)


# # Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_testage_1, y_testage_1)
y_predage_1 = gaussian.predict(x_valage_1)
acc_gaussianage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_gaussianage_1)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbkage_1, acc_sgdage_1, acc_knnage_1, acc_randomforestage_1, 
              acc_decisiontreeage_1, acc_perceptronage_1, acc_linear_svcage_1, acc_svcage_1, acc_logregage_1, 
               acc_gaussianage_1]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


predictions = gbk.predict(x_tst_test_age.drop('PassengerId', axis=1))


# In[ ]:


p=0
for i in range(418):
    if np.isnan(test_test_age['AgeGroup'][i]) == True:
        test_test_age['AgeGroup'][i] = predictions[p]
        p+=1


# In[ ]:


from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.05, random_state = 0)


# # Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# # Stochastic Gradient Descent

# In[ ]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# # KNN or k-Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# # Perceptron

# In[ ]:


from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# # Linear SVC

# In[ ]:


from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# # Support Vector Machines

# In[ ]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# # Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[ ]:


models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbk, acc_sgd, acc_knn, acc_randomforest, acc_decisiontree, acc_perceptron, acc_linear_svc, acc_svc, acc_logreg, 
               acc_gaussian]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = knn.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission_knn.csv', index=False)


# # Sequential NN

# In[ ]:


x_tr = train.iloc[:,2:].as_matrix()
y_tr = train.iloc[:,1].as_matrix()


# In[ ]:


X_test = test.iloc[:,1:].as_matrix()


# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(8,)))

model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu))

# model.add(tf.keras.layers.Flatten(input_shape=(8,)))

model.add(tf.keras.layers.Dense(2,activation = tf.nn.softmax))


# In[ ]:


model.summary()


# In[ ]:


# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])


# In[ ]:


model.fit(x_tr, y_tr, batch_size=128, epochs = 60)


# In[ ]:


predictions = model.predict([X_test])


# In[ ]:


import csv
data = [['PassengerId', 'Survived']]
for i in range(1,419):
    data.append([i+891,np.argmax(predictions[i-1])])
print(data)
with open('submission_own_NN.csv','w',newline='') as fp:
    a=csv.writer(fp,delimiter = ',')
    a.writerows(data)


# In[ ]:




