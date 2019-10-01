#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import numpy as np 
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt 
get_ipython().magic(u'matplotlib inline')
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

#import project files
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#Any results you write to the current directory are saved as output.


# In[ ]:


#Use head of data as preview, get info of training and test data
print("\n\nTop of the training data:")
print(train.head())

print("\n\nInfo summary of training data")
print(train.info())

print("\n\nInfo summary of test data")
print(test.info())


# Data Simplification and Dealing with Missing Values

# In[ ]:


#fill fare value that is missing from test data with median 
test["Fare"].fillna(test["Fare"].median(), inplace=True) ;

#visualise distribution of fares
plt.hist(train['Fare'], bins=30, range=[0,300]) #number of bins chosen using the square root rule
plt.title('Distribution of Fares')
plt.xlabel('Fare')
plt.ylabel('Quantity')
plt.show()
plt.clf()

#split fares into 3 classes and assign integer values: high(2), medium(1), low(0)
groups = [0, 1, 2]
bins = [-1, np.percentile(train['Fare'], 33), np.percentile(train['Fare'], 67), max(train['Fare'])+1]
train['Fare'] = pd.cut(train['Fare'], bins, labels=groups)

#same for test data
bins = [-1, np.percentile(test['Fare'], 33), np.percentile(test['Fare'], 67), max(test['Fare'])+1]
test['Fare'] = pd.cut(test['Fare'], bins, labels=groups)#Passengers' names, tickets, IDs not relevant to survival; due to sparsity I will also drop the cabin variable.
passIdstacking = train['PassengerId']
train = train.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1) #cabin may be worth revisiting later
test = test.drop(['Name', 'Ticket', 'Cabin'], axis=1) #Keep PassengerId here for prediction purposes
("")


# In[ ]:


#find most common Embarked value to use to fill the 2 NA values:
print(train['Embarked'].value_counts())

#since the majority of passengers boarded at Southampton, we fill missing data with this value.
train['Embarked'].fillna('S', inplace=True)

#convert emmarked values to integers; S = 0, C = 1, Q = 2 
train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2

test['Embarked'][test['Embarked'] == 'S'] = 0
test['Embarked'][test['Embarked'] == 'C'] = 1
test['Embarked'][test['Embarked'] == 'Q'] = 2


# In[ ]:


#replace missing age values with median corresponding to the sex and far category of the passenger

for i in range(3):
    train['Age'].loc[(train['Sex'] == 'male') & (train['Fare'] == i) & pd.isnull(train['Age'])] = train['Age'].loc[(train['Sex'] == 'male') & (train['Fare'] == i)].median()
    train['Age'].loc[(train['Sex'] == 'female') & (train['Fare'] == i) & pd.isnull(train['Age'])] = train['Age'].loc[(train['Sex'] == 'female') & (train['Fare'] == i)].median()
    test['Age'].loc[(test['Sex'] == 'male') & (test['Fare'] == i) & pd.isnull(test['Age'])] = test['Age'].loc[(test['Sex'] == 'male') & (test['Fare'] == i)].median()
    test['Age'].loc[(test['Sex'] == 'female') & (test['Fare'] == i) & pd.isnull(test['Age'])] = test['Age'].loc[(test['Sex'] == 'female') & (test['Fare'] == i)].median()
    
print(train.info())
print(test.info())


# In[ ]:


#put ages into categories by decade

train['AgeStat'] = float('NaN')
test['AgeStat'] = float('NaN')

for i in range(10):
    train['AgeStat'].loc[(train['Age'] <= ((10*i) + 10)) & (train['Age'] > (10*i))] = i+1
    test['AgeStat'].loc[(test['Age'] <= ((10*i) + 10)) & (test['Age'] > (10*i))] = i+1

print(train['AgeStat'].value_counts())
print(test['AgeStat'].value_counts())


#can now drop Age column and feed age classes into prediction algorithms
train = train.drop('Age', axis=1)
test = test.drop('Age', axis=1)


# In[ ]:


#create new column specifying if passenger has family on board the Titanic 
# this allows us to drop the Parch and SibSp columns 
train['Family'] = train['Parch'] + train['SibSp']
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] = test['Parch'] + test['SibSp']
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0

train = train.drop(['Parch', 'SibSp'], axis=1)
test = test.drop(['Parch', 'SibSp'], axis=1)

print(train.info())
print(test.info())


# In[ ]:


#convert non-integer data to integers as required by sk learn functions
train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1
test['Sex'][test['Sex'] == 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1

#train['AgeStat'] = train['AgeStat'].astype(int)
#test['AgeStat'] = test['AgeStat'].astype(int)

train['Sex'] = train['Sex'].astype(int)
test['Sex'] = test['Sex'].astype(int)

train['Embarked'] = train['Embarked'].astype(int)
test['Embarked'] = test['Embarked'].astype(int)

train['Fare'] = train['Fare'].astype(int)
test['Fare'] = test['Fare'].astype(int)

print(train.info())
print(test.info())


# Exploration (Survival Rates):

# In[ ]:


#calculate survival rates for all passengers 
print("Survival rate across all passengers (1 => Survived, 0 => not survived): ")
print(train['Survived'].value_counts(normalize=True))

print("\n\nProportion of Passengers on board by gender")
print(train['Sex'].value_counts(normalize=True))


# In[ ]:


#Survival Rates by Age

for i in range(10):
     print("\n\nSurvival rates for age category: ", i)
     print(train['Survived'][train['AgeStat'] == i].value_counts(normalize=True))


# In[ ]:


#Survival Rates by Gender

print("\n\nSurvival rates for males: ")
print(train['Survived'][train['Sex'] == 0].value_counts(normalize=True))

print("\n\nSurvival rates for females: ")
print(train['Survived'][train['Sex'] == 1].value_counts(normalize=True))

#it seems that females are more likely to survive than males 


# In[ ]:


#Survival Rates due to family on board

print("\n\nSurvival rates among those who have family on board: ")
print(train['Survived'][train['Family'] == 1].value_counts(normalize=True))

print("\n\nSurvival rates among those with no family on board")
print(train['Survived'][train['Family'] == 0].value_counts(normalize=True))

#those with family on board are more likely to survive


# In[ ]:


#survival rates due to passenger class; First(1), Second(2), Third(3)

print('Survival rate of those in First Class: ')
print(train['Survived'][train['Pclass'] == 1].value_counts(normalize=True))

print('\n\nSurvival rate of those in Second Class: ')
print(train['Survived'][train['Pclass'] == 2].value_counts(normalize=True))

print('\n\nSurvival rate of those in Third Class: ')
print(train['Survived'][train['Pclass'] == 3].value_counts(normalize=True))


# In[ ]:


#Survival rates due to fare class

print("Survival rate of those who paid low fares: ")
print(train['Survived'][train['Fare'] == 0].value_counts(normalize=True))

print("\n\nSurvival rate of those who paid medium fares: ")
print(train['Survived'][train['Fare'] == 1].value_counts(normalize=True))

print("\n\nSurvival rate of those who paid high fares: ")
print(train['Survived'][train['Fare'] == 2].value_counts(normalize=True))

#those who had paid more to board the ship were more likely to survive


# Analysis and Prediction:

# In[ ]:


#Adjust data to form required by sk-learn algorithms 
trainingX = train.drop('Survived', axis=1)
trainingY = train['Survived']
testX = test.drop('PassengerId', axis=1).copy()


# In[ ]:


#prediction using random forests
rf = RandomForestClassifier(n_estimators=100)
rf.fit(trainingX, trainingY)
rfprediction = rf.predict(testX)
print(rf.feature_importances_)
print(rf.score(trainingX, trainingY))


# In[ ]:


# prediction using logistic regression
#lr = LogisticRegression()
#lr.fit(trainingX, trainingY)
#lrprediction = lr.predict(testX)
#print(lr.score(trainingX, trainingY))


# In[ ]:


#prediction using KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(trainingX, trainingY)
knnprediction = knn.predict(testX)
print(knn.score(trainingX, trainingY))


# In[ ]:


#prediction using gradient boosting

gb = GradientBoostingClassifier(n_estimators = 200)
gb.fit(trainingX, trainingY)
gbprediction = gb.predict(testX)
print(gb.score(trainingX, trainingY))


# In[ ]:


#ensemble: bagging
#voting system
prediction = rfprediction + gbprediction + knnprediction
prediction[prediction == 1] = 0
prediction[prediction == 2] = 1
prediction[prediction == 3] = 1


# In[ ]:


#create solution csv for submission
PassengerId = np.array(test['PassengerId']).astype(int)
solution = pd.DataFrame(prediction, PassengerId, columns=['Survived'])
solution = solution.to_csv('newsolution', index_label = ['PassengerId'])


# next steps: 1) Use XGBoost instead of GB 
#                     2) apply stacking on initial predictions 
#                     3)cross-validation to improve choosing of parameters
