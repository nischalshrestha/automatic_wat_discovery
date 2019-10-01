#!/usr/bin/env python
# coding: utf-8

# *Titanic Disaster: Train, Trick & Rewrite History*

# # **Introduction:**
# 
# This is my first kernel submission. This kernel is trained on the given dataset and uses Decision Trees to predict the Survival of any passenger.  And as the title suggests, if the classifier predicts a passenger did not survive, then I have used a plain brute force method to look through the combination of the features that the passenger would have had a possiblity of changing (Fare, Ticket Class & Port of Embarkation)  to find the first best chance of Survival. In short, I call it '*A Dumb Titanic Survival Guide*'. Please note that the data clean up and  correlation of features have been used from this [tutorial](http://https://www.kaggle.com/startupsci/titanic-data-science-solutions).
# 
# To begin with I have split the code into two parts:
# 
# **Part 1 **- This part contains three steps to the code to load the data, perform workflow goals as mentioned in the tutorial above and train a decision tree classifier and calculate it's accuracy.
# 
# **Part 2** - In this part I pick a random test data and predict the Survival using the classifier trained in step 3.  If the passenger did not survive, then I try to find the first best combination of features (Fare, Ticket Class & Port of Embarkation) for that passenger that increases his/her chance of survivial. 
# 
# Please run this kernel and leave your comments/suggestions to improve. 
# 
# Thanks.
# 
# Pranesh

# ###  **Part 1: ** 
# **Step 1: ** Load the training and test data for the given datasets.

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import time as tm
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]

#print(train_df.columns.values)
print('Training & test data loaded.')


# **Step 2:**  Perform workflow goals as described in this [tutorial](https://www.kaggle.com/pashern/titanic-data-science-solutions).

# In[ ]:


train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

    
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_df.Embarked.dropna().mode()[0]

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

print('Workflow goals performed.')


# **Step 3:**  Use Decision Tree classifer  to train the data and test it on the test set.  The accuracy is about 86.76%

# In[ ]:


#create a decision tree classifer
decision_tree = DecisionTreeClassifier()

#train the classifier using the training set
decision_tree.fit(X_train, Y_train)

#predict the test set using the classifier
Y_pred = decision_tree.predict(X_test)

#calclualte the accuracy on the classifier
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('The accuracy for the given training data using Decision Tree classifier is: {0:.2f} %' 
      .format(acc_decision_tree))


# In[ ]:


#submission steps
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('mysubmission.csv', index=False)


# ###  **Part 2: ** 
# 
# Pick a random test data  and try to preict the survival for that passenger.  If the passenger did not survive, then loop through the three freatures Fare, Ticket Class & Port of Embarkation to find the first best combination for the classifier that would predict that the passenger survived.  The feature values (Pclass_list, Fare_list, Embarked_list) in the below code are ordered in the decreasing order of correlation with survival.   This was learnt from step 2 in Part 1.  I have used dictionaries to reverse the converting effect on the data performed again in step 2 in Part 1.  

# In[ ]:


#Feature reversal dictionaries
Fare_dict = {0:"around £7.5", 1:"between £7.5 & £14.5", 2:"between £14.5 & £31", 3:"more than £31"}
Embarked_dict = {0:"Southampton", 1:"Cherbourg", 2:"Queenstown"}
Pclass_dict = {1:"upper", 2:"middle", 3:"lower"}
Sex_dict = {1:'female', 0:'male'}
Pronoun_dict = {1:"She", 0:"He"}
Age_dict = {0:"below 16", 1:"between 16 & 32", 2:"between 32 & 48", 3:"between 48 & 64", 4:"older than 64"}
Survive_dict = {0:"unfortunately did not make it alive :(",
                1:"miraculously survived and lived happily there after :)"}

#Function that tries to predict tbe best combination to help the passenger survive. 
def possibs(curr_row_in):
    curr_row = curr_row_in
    #print((curr_row))
    pclass = curr_row[0][0]
    sex = curr_row[0][1]
    age = curr_row[0][2]
    fare = curr_row[0][3]
    embarked = curr_row[0][4]
    title = curr_row[0][5]
    isalone = curr_row[0][6]
    ageclass = curr_row[0][7]
    Pclass_list = [1,2,3]
    Fare_list = [2,3,1,0]
    Embarked_list = [1,2,0]
    #count = 0
    
    for pc in Pclass_list:
        for f in Fare_list:
            for em in Embarked_list: 
                #count = count + 1
                possibs_row = [pc, sex, age, f, em, title, isalone, ageclass]
                df = pd.DataFrame([possibs_row])
                #print('Count:', count)
                survival = decision_tree.predict(df)
                #print(pc,f,em,survival)
                if survival == 1:
                    return ((pc,f,em))


# Pick a random row from the test set.
row, col = X_test.shape
dom =  np.random.randint(0, high=row, size=1)
Y_pred = decision_tree.predict(X_test.loc[dom])

#convert the random row to a list and assign 
curr_row = X_test.loc[dom].values.flatten().tolist()

#unpack the list to variables
pclass = curr_row[0]
sex = curr_row[1]
age = curr_row[2]
fare = curr_row[3]
embarked = curr_row[4]
title = curr_row[5]
isalone = curr_row[6]
ageclass = curr_row[7]


#Predict the survival of the random passenger
print('\nWe are going to predict the survival of a {0:s} passenger, who was aged {1:s}. {2:s} embarked at {3:s} and paid a fare {4:s} to get a {5:s} class ticket. The passenger {6:s}'
      .format(Sex_dict[sex], Age_dict[age], Pronoun_dict[sex], Embarked_dict[embarked], Fare_dict[fare], Pclass_dict[pclass], Survive_dict[Y_pred[0]]))

#If the passenger did not survive, then call the function to get the best feautre combinations for survvial.
if(Y_pred[0]!=1):
    #function call
    (pc,f,em) = possibs([curr_row])
    
    #outputs
    print('\n\nSince the passenger did not survive the tragedy, now let\'s try to use \'The Dumb Titanic Survival Guide\' to predict the best options that would have helped the passenger survive.' )
    print('\n\npredicting...\n\n')
    tm.sleep(2)
    print('The Dumb Titanic Survival Guide says...\n')
    tm.sleep(1)
    print('The passenger would have had a better chance of survival if {0:s} had purchased {1:s} class ticket for a price {2:s} and embarked at {3:s}'
         .format(Pronoun_dict[sex], Pclass_dict[pc], Fare_dict[f], Embarked_dict[em]))

###THE END###


# *Thank you for trying this kernel.  Please leave your valuable feedback in the comments and upvote if you liked the appach. *
