#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Any results you write to the current directory are saved as output.


# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data= pd.read_csv("../input/test.csv")


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# In[ ]:


# there is missing values in both datasets.
# there is no "Survived" column in train_data because that is what we have to predict!  
# my main aim is to find the "Survived" value for each Passenge


# In[ ]:


# Before going there, let's analyse and visualise our data to get a feel of it.
# I need only useful features to be able to predict efficiently.
# Let's start from the first column^
# PassengerId: It is clearly of no use; just a serial no. Let's DROP it then. 
train_data.drop(['PassengerId'], axis=1, inplace=True)


# In[ ]:


# Let's move on to the next feature 'Name'
# Useless feature quite obviously. 
# Let's drop it
train_data.drop(['Name'], axis=1, inplace=True)
train_data.head()


# In[ ]:


# "Survived" == 0 indicates "DID NOT Survive"; 1 == "Survived"
# Now, we've looked at features uptil Pclass; Next is "Sex"


# In[ ]:


# There are many children, so let's study them separately.
# Convert "Sex" into "Person" column which can take values: "Male", "Female", "Child"
# Let's create a function for that
def what_person(passenger):
    age,sex = passenger
    if age <= 16:
        return 'Child'
    else: 
        return sex


# In[ ]:


# Let's "apply" now
train_data["Person"] = train_data[['Age','Sex']].apply(what_person, axis=1)
# axis=1 specifies that the operation is to be done on columns!
# Drop "Sex" now, since it is redundant
train_data.drop(['Sex'], axis=1, inplace=True)
train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


print("Missing Age values:", train_data['Age'].isnull().sum())


# In[ ]:


# Let's fill the missing^ Age values now
# Generate random numbers between mean-std & mean+std
mean = train_data['Age'].mean()
std = train_data['Age'].std()

r = np.random.randint(mean-std, mean+std)
train_data["Age"].fillna(r, inplace=True)

train_data.info()


# In[ ]:


# Let's look at next two features:
# SibSp is any siblings/spouses on board?
# Parch is any parent/child on board?
# We could reduce these to a single feature: "WithFamily"?
# This would make our feature-vector more efficient and dimensionality reduction!!
train_data['WithFamily'] =train_data['SibSp'] + train_data['Parch']
train_data.drop(['SibSp','Parch'], axis=1, inplace=True)
train_data.head(10)


# In[ ]:


# Let's clean that!
# If "WithFamily" == 0, He was alone. Hence, value should be 0.
train_data['WithFamily'].loc[train_data['WithFamily'] > 0] = 1
train_data.head(10)


# In[ ]:


# Next feature is Ticket, which is useless again.lets Remove it!
train_data.drop(['Ticket'], axis=1, inplace=True)


# In[ ]:


test_data.info()


# In[ ]:


# Fare:
# Missing values only in test_df
test_data["Fare"].fillna(test_data["Fare"].median(), inplace=True)


# In[ ]:


# Convert from float to int
train_data['Fare'] = train_data['Fare'].astype(int)
test_data['Fare'] = test_data['Fare'].astype(int)


# In[ ]:


# Let's see if they vary with Survival chances
fare_notSurvived = train_data["Fare"][train_data["Survived"] == 0]
fare_survived =train_data['Fare'][train_data["Survived"] == 1]
print("Died: ", fare_notSurvived.mean())
print("Survived: ", fare_survived.mean())


# In[ ]:


train_data.head()


# In[ ]:



# Now, I've looked at "Survived" "Pclass" "Age" "Fare"# Now, w 
# Created two new features/columns "Person" "WithFamily"; also dropped some columns 
# Let's look at Cabin now:


# In[ ]:


# Cabin is in the format: C85 where the first letter ('C', in this case) is the deck
# Deck seems to give out important info as compared to the room no. 
# Let's extract all decks from Cabin; let's drop null values first!
deck = train_data['Cabin'].dropna()
deck.head()


# In[ ]:


floor = []
for level in deck:
    floor.append(level[0])

# To visualise it, let's convert it into a DataFrame
df = pd.DataFrame(floor, columns=['Level'])


# In[ ]:


train_data.info()


# In[ ]:


#  the 'Cabin' column has a lot of missing values.
# On top of that, there is just one value for deck 'T' which doesn't make a lot of sense.
# Filling 75% of the values on our own would affect prediction
# Hence, it is better to drop this column
train_data.drop('Cabin', axis=1, inplace=True)
train_data.head()


# In[ ]:


train_data.info()


# In[ ]:


# Just two missing values! Let's fill it with "S" (the most frequent)# Just t 
train_data['Embarked'].fillna("S", inplace=True)


# In[ ]:


# Passengers that embarked at "S" had a less rate of survival; Let's confirm that:
embark = train_data[['Embarked', 'Survived']].groupby(['Embarked']).mean()
embark


# In[ ]:


# Let's make our test_data compatible with train_data; since we're going to train our classifier on train_data


# In[ ]:


test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Now, let's create Person for test_df:
test_data["Person"] =test_data[['Age','Sex']].apply(what_person, axis=1)
test_data.drop(['Sex'], inplace=True, axis=1)

# Now, let's create WithFamily for test_df:
test_data['WithFamily'] = test_data['SibSp'] + test_data['Parch']
test_data.drop(['SibSp','Parch'], axis=1, inplace=True)
test_data['WithFamily'].loc[test_data['WithFamily'] > 0] = 1


# In[ ]:


test_data.info()


# In[ ]:


print("Missing: ", test_data['Age'].isnull().sum())


# In[ ]:


# Let's fill in the missing Age values
mean = test_data['Age'].mean()
std = test_data['Age'].std()

r = np.random.randint(mean-std, mean+std)
test_data['Age'].fillna(r, inplace=True)

# Change its dataype to int
train_data['Age'] =train_data['Age'].astype(int)
test_data['Age'] = test_data['Age'].astype(int)


# In[ ]:


test_data.info()


# In[ ]:


# There is one last issue remaining before i can feed this dataset to ML algortihm
# Embarked & Person need to converted to Numeric variables
# I'll use dummy variables: 
# It is a variable that takes 0/1 indicating absence/presence of a particular category
# You can read more about it - https://en.wikipedia.org/wiki/Dummy_variable_(statistics)

# EMBARKED-
titanic_embarked = pd.get_dummies(train_data['Embarked'])
titanic_embarked.head()


# In[ ]:


train_data =train_data.join(titanic_embarked)
train_data.head()


# In[ ]:


# Person 
titanic_person = pd.get_dummies(train_data['Person'])
titanic_person.head()


# In[ ]:


train_data = train_data.join(titanic_person)
# Let's remove Person/Embarked now
train_data.drop(['Person','Embarked'], axis=1, inplace=True)
train_data.head()


# In[ ]:


# Let's repeat the same procedure for test_data# Let's  
test_embarked = pd.get_dummies(test_data['Embarked'])
test_data = test_data.join(test_embarked)

test_person = pd.get_dummies(test_data['Person'])
test_data = test_data.join(test_person)

test_data.drop(['Person','Embarked'], axis=1, inplace=True)
test_data.head()


# In[ ]:


# Now is the time set up our training and test datasets:
x_train = train_data.drop(['Survived'], axis=1)

y_train = train_data['Survived']

x_test = test_data.drop(['PassengerId'], axis=1)

x_train.head()


# In[ ]:


from sklearn import svm


# In[ ]:


model = svm.SVC(kernel='linear', C=1, gamma=1) 


# In[ ]:


model.fit(x_train, y_train)


# In[ ]:


prediction = model.predict(x_test)


# In[ ]:


prediction


# In[ ]:


model.score(x_train, y_train)


# In[ ]:


# Let's finally submit !!!!
sub_file = pd.DataFrame({'PassengerId':test_data['PassengerId'], 'Survived':prediction})
sub_file.head()


# In[ ]:


sub_file.to_csv('result.csv', index=False)


# In[ ]:




