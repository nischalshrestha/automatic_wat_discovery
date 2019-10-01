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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#machine learning imports for determining predictions
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
#remember that this line of code ensures that everything you do to the training data, you also do
#to the test data when you write in terms of combine.


# Making sure that any for loop can just be ran for combine instead of doing it for both of the dataset

# In[ ]:


train_df.info()
print(''*40)
test_df.info()


# This line of code displays the amount of data that the dataset actually has. It tells me which columns have data for everything, and which columns are missing data

# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# Taking Manav Seghal's code to drop three sections that I think have nothing to do with the code, even without correlating it with who actually survived and died in the end, because I think class has more to do with the death or the survival than the intricate little bits of the ticket number the cabin number, and the specific name or title of the person

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].replace('S', 0)
    dataset['Embarked'] = dataset['Embarked'].replace('C', 1)
    dataset['Embarked'] = dataset['Embarked'].replace('Q', 2)
    dataset['Embarked'] = dataset['Embarked'].fillna(value=3)
    
train_df.head()


# **Analysis on last set of code**
# In the last bit of code that I wrote, I stole the code from the other guy's data (manav) and used it to see what the embarked part would end up looking like when switching it from categorical to numerical. I see 1's and 0's on the embarked section, which tells me that my code worked effectively

# In[ ]:


for dataset in combine:
    dataset['Age'] = dataset['Age'].fillna(value=35)


# For loop to replace missing values of age with a random number, this time it is an age of 35, because I ran it multiple times with different ages to see which age gave me the best prediction rate, and I settled on 35 as my final substitution value.

# In[ ]:


for dataset in combine:
    dataset['Fare'] = dataset['Fare'].fillna(value=200)
    
#train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False). mean().sort_values(by='Survived', ascending=False)


# same thing as a for loop with the fares of everyone's missing data, to see which ones would give better data and which ones would be worse. It doesn't really matter what the value is, because the original data for the section I'm fixing really has no issues.

# In[ ]:


for dataset in combine: 
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0}).astype(int)


# I still need to make sure every column is numerical to ensure that the regressions actually work correctly. This time I switched female to one and male to zero so that there is no confusion on that.

# In[ ]:


#running tests to determine which values and columns are important.
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False). mean().sort_values(by='Survived', ascending=False)


# From the code above, it is obvious that there is a correlation between aclass and survival rate. This is something to keep in mind, that class plays a big part in determining survival rate. Now i want to test for gender,to see how the two main things would correlate to determining survival rate

# In[ ]:



train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False). mean().sort_values(by='Survived', ascending=False)


# **Analysis**
# This last section shows an obvious correlation between survival rate and gender. This section acts as a pearson correlation coefficient because testing for the actual results is very similar to determining a correlation coefficient. Based off of the fact that females had an almost 75% chance of survival, compared to a 19% survival rate of males shows that this feature needs to be included as a part of predicting the final percentages of those that survived and those who did not.

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
X_train = X_train.drop("PassengerId", axis=1).copy()
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# The last bit of code below is just running all of the classifiers to see what gives me the best fit predictions.

# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# I just copied and pasted all the prediction code from the workbook we extracted our data from. After putting all of them in, these are the final regressions and prediction values that I got below. I think that my data is a bit overfit for the random forest and decision tree because I decided to cut out a lot of the data, just because I didn't think it was necessary. This made it so that a lot of the data could have been overfit, or that not all of them had enough data to form good predictions in the first place.

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission.csv', index=False)

