#!/usr/bin/env python
# coding: utf-8

# # ** I have made some significant change to my method of Feature Selection. This can been seen in my other kernel: My titanic attempt:) Improved Feature Selection**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[ ]:


#Import the necessary data
test_df = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')
combine = [train_df, test_df]


# In[ ]:


#determine the quality of the provided data in order to determine the proceedure for data cleaning.
#missing values
#train_df.info()
print ("percentage of missing values for age: ",(1-((714/891)))*100)
print ("Percentage of missing values for cabin: ",(1-((204/891)))*100)


# Based on the results above, I need to consider if there is any point in cleaning the data, ie, replacing missing values.
# 
# 1.Cabin:
# Every passenger would have been assigned a cabin and yet this information has not been kept. This cannot be considered a random miss. Instead, this suggests it is poor record keeping. The next thing to consider is the volume missing data. A massive 77.10% is missing, if I were to deal with this, I risk seriously skewing the data. If I remove the rows that are missing this data, I lose 77% of my data overall, this is not an option. If I replace it, I have very little information in order to determine the best value to replace the missing value. I suggest the best option is to ignore this column completely. As much as it would be interesting to determine if a cabins location has a bearing on survival, there just isn't enough data for the results to be considered reliable.
# 
# 2.Age:
# Intuitively I would think age has a bearing on survival. it is well known that when it comes to the priority of saving a person, it is women and children first, therefore is a person is young, such as a child, they have a higher chance of survival, compare to an adult. The percentage of missing values is 19.86%. It is reasonable to suggest this data can be cleaned. 
# As to the reason for it not being recorded, the person in question may not have wanted their age to be known, they were a child and it was missed, or they just didn't know. For the moment I would like to keep the variable in, therefore I will replace the missing value. The below suggests that the data is slightly skewed, I will use the mode as althought the mean is arond 29 years, the mode suggests there were many younger people on board.
# 
# 3. Other variables that add no value:
# There are some columns in there that will not add any value and will instead just create noise. I will fix the above 2 issues and then determine the columns that add no value.
# 
# Conclusion:
# I will remove the Cabin variable and replace the missing age values with the mode. 

# In[ ]:





# In[ ]:


print ("The average age: ",train_df["Age"].mean())
print ("The mode of age in train dataset: ",train_df["Age"].mode())
print ("The mode of age in test dataset: ",test_df["Age"].mode())


# In[ ]:


#remove the cabin column
train_df.drop('Cabin', axis = 1, inplace = True)
test_df.drop('Cabin', axis = 1, inplace = True)
#while we are here, remove the name & tickets columns as it is obvious it will add no value
train_df.drop(['Name','Ticket'], axis = 1, inplace = True)
test_df.drop(['Name','Ticket'], axis = 1, inplace = True)
combine = [train_df,test_df]


# In[ ]:


train_df['Age'].fillna(24, inplace=True)
test_df['Age'].fillna(21, inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


#Turn the sex category into a binary variable
gender_mapping = {"female": 0, "male": 1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(gender_mapping)
    dataset['Sex'] = dataset['Sex'].fillna(0)

train_df.head()


# In[ ]:





# The above correlaiton matrix highlights the various levels off correlations between survival and the other available variables.
# Considering statistically, the variables that have an important level of correlation are:
# * Pclass - this suggests a weak negative relationship between class and the ability to survive
# * Sex - This suggests a moderate negative relationship
# * Fare - This suggest a weak positive relationship.
# 
# The above results are somewhat surprising as I would have expected age to have more of an impact. The theory I have, which is seen across many data sets is the values each variable gets. Fare and Age are large numbers when compared to passenger class, or gender, therefore it is a good idea if I now fix up some of the other data and consider grouping these variables.

# In[ ]:


#split the ages up into groups
#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
#train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


#for dataset in combine:    
 #   dataset.loc[ dataset['Age'] <= 16.336, 'Age'] = 0
 #   dataset.loc[(dataset['Age'] > 16.336) & (dataset['Age'] <= 32.252), 'Age'] = 1
 #   dataset.loc[(dataset['Age'] > 32.252) & (dataset['Age'] <= 48.168), 'Age'] = 2
 #   dataset.loc[(dataset['Age'] > 48.168) & (dataset['Age'] <= 64.084), 'Age'] = 3
 #   dataset.loc[ dataset['Age'] > 64.084, 'Age']
 #   dataset['Age'] = dataset['Age'].astype(int)
#train_df.head()


# In[ ]:





# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[ ]:


#now banding for fare
#train_df['FareBand'] = pd.cut(train_df['Fare'], 4)
#train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


#for dataset in combine:    
#    dataset.loc[ dataset['Fare'] <= 128.082, 'Fare'] = 0
#    dataset.loc[(dataset['Fare'] > 128.082) & (dataset['Fare'] <= 256.165), 'Fare'] = 1
#    dataset.loc[(dataset['Fare'] > 256.165) & (dataset['Fare'] <= 384.247), 'Fare'] = 2
#    dataset.loc[ dataset['Fare'] > 384.247, 'Fare'] = 3
#    dataset['Fare'] = dataset['Fare'].astype(int)
#train_df.head()


# In[ ]:


#drop the 2 band column
#train_df = train_df.drop(['FareBand','AgeBand'], axis=1)
#combine = [train_df, test_df]


# In[ ]:


#Need to turn embarked into a number
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[ ]:


train_df.corr(method = 'pearson')


# In[ ]:


#Taking the above correlation matrix into account, remove PassengerID, Age, Sibsp, Parch
train_df = train_df.drop(['PassengerId','Age','SibSp','Parch','Fare'], axis=1)
combine = [train_df, test_df]


# In[ ]:


train_df.corr(method = 'pearson')


# In[ ]:


#set up the data for the models
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(['PassengerId','Age','SibSp','Parch','Fare'], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
#print (X_train.head())


# In[ ]:


test = SelectKBest(score_func=chi2, k="all")
fit = test.fit(X_train, Y_train)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X_train)
# summarize selected features
print(features[0:5,:])


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
Y_predSVM = svc.predict(X_test)
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
Y_predRF = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


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
        "Survived": Y_predSVM
    })

#submission.head()
submission.to_csv('submission.csv', index=False)


# In[ ]:




