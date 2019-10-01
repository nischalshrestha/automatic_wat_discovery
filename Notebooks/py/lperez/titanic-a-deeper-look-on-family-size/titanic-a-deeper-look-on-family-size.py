#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Several notebooks explain that family size has a great impact on survival rate on the Titanic. Travelling alone decreases significantly the chances of survival. This is true.
# 
# But it does not mean that adding an "Alone" feature would help making a better prediction.
# 
# Indeed, the reason why being alone decreases the survival rate is because being alone increases the chances of being a man. And on the Titanic, the rule was "children and women first".
# 
# In this notebook, I demonstrate that it is worth adding a "Large Family" feature, but it is counter-productive to add an "Alone" feature in our prediction dataset. I also try to explain why.
# 
# This is my first work on Kaggle. This notebook has been forked from [Titanic best working classifier by Sina][1].
# 
# Of course, comments on this work are more than welcome!
# 
#   [1]: https://www.kaggle.com/sinakhorami/titanic-best-working-classifier

# # Reading data

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]

print (train.info())


# # Feature Engineering
# 
# In this notebook, we are going to look only at family size, and to other features related to families: age and sex.

# ## SibSp and Parch
# 
#  - SibSp: number of siblings/spouse
#  
#  - Parch: number of childre/parents
# 
# With the number of siblings/spouse and the number of children/parents we can create new feature called Family Size.

# In[ ]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
pd.crosstab(train['FamilySize'], train['Survived']).plot(kind='bar', stacked=True, title="Survived by family size")
pd.crosstab(train['FamilySize'], train['Survived'], normalize='index').plot(kind='bar', stacked=True, title="Survived by family size (%)")


# It seems that for families from 1 to 4 people, family size increases survival rates. But for families of 5 and up, survival rates is much lower.

# ## Sex and family size
# Let's split our dataset according to Sex feature and see what's happening for different family sizes.

# In[ ]:


female = train[train['Sex'] == 'female']
male = train[train['Sex'] == 'male']
 
# Total number
fig, [ax1, ax2] = plt.subplots(1,2, sharey=True)
fig.set_figwidth(12)
pd.crosstab(female['FamilySize'], female['Survived']).plot(kind='bar', stacked=True, title="Female", ax=ax1)
pd.crosstab(male['FamilySize'], male['Survived']).plot(kind='bar', stacked=True, title="Male", ax=ax2)

# Percentage
fig, [ax1, ax2] = plt.subplots(1,2)
fig.set_figwidth(12)
pd.crosstab(female['FamilySize'], female['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Female", ax=ax1)
pd.crosstab(male['FamilySize'], male['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Male", ax=ax2)


# This is very interesting.
# 
# First of all, we see that for both sex, family sizes of 5 and up lead to low survival rates.
# 
# For females in families up to 4, the survival rate is about 80%, regardless of family size. 
# 
# For males in families up to 4, the survival rate increases with family size. Let's see how this effect is related to age.

# ## Male kids vs adults

# In[ ]:


kidsmale = male[male['Age'] < 15]
adultsmale = male[male['Age'] >=15 ]

print ("Number of male kids: ")
print (kidsmale.groupby(['FamilySize']).size())
print ("")
print ("Number of male adults: ")
print (adultsmale.groupby(['FamilySize']).size())

# Size of samples
fig, [ax1, ax2] = plt.subplots(1,2)
fig.set_figwidth(12)
sns.countplot(x='FamilySize', data=kidsmale, ax=ax1)
ax1.set_title('Number of male kids')
sns.countplot(x='FamilySize', data=adultsmale, ax=ax2)
ax2.set_title('Number of male adults')

# Percentage
fig, [ax1, ax2] = plt.subplots(1,2)
fig.set_figwidth(12)
pd.crosstab(kidsmale['FamilySize'], kidsmale['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Kids male", ax=ax1)
pd.crosstab(adultsmale['FamilySize'], adultsmale['Survived'], normalize = 'index').plot(kind='bar', stacked=True, title="Adults male", ax=ax2)


# This is even more interesting. For males from families up to 4 people, we can see here that there is no real impact of family size. Indeed, in these families, almost all the male kids have survived, and the survival rate of male adults does not change with family size.
# 
# ## Conclusion on family size
# According to the previous analysis, we can make the following assumption: for families up to 4 people, the impact of family size on the survival rate can be explained by age and sex.
# 
# Therefore, we don't need to create features like IsAlone. Let's try three different resolutions and see which one is the best:
# 
# 1. Without family size
# 
# 2. With a LargeFamilies feature (up to 4 / 5 and more)
# 
# 3. With a three classes FamilySize feature (alone, 2-4, 5 and more)

# # Preparing data for resolution
# 
# ## Family size categories
# Let's create LargeFamily (2 classes) and FamilyClass (3 classes) features.

# In[ ]:


for dataset in full_data:
    dataset['LargeFamily'] = dataset['FamilySize'].apply(lambda r: 0 if r<=4 else 1)
    
    dataset.loc[ dataset['FamilySize'] == 1, 'FamilyClass'] = 0
    dataset.loc[ (dataset['FamilySize'] <= 4) & (dataset['FamilySize'] > 1), 'FamilyClass'] = 1
    dataset.loc[ dataset['FamilySize'] >= 5, 'FamilyClass'] = 2
    dataset['FamilyClass'] = dataset['FamilyClass'].astype(int)


# 
# 
# ## Names ##
# inside this feature we can find the title of people.

# In[ ]:


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))


#  so we have titles. let's categorize it and check the title impact on survival rate.

# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# # Other data
# Now let's clean all other fields and map our features into numerical values.

# In[ ]:


for dataset in full_data:   
    # Fill missing values in Embarked with most frequent port 'S'
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
    # Fill missing values in Fare with median
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    # Fill missing values in age with random data based on mean and standard variation
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
        
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
    
# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)

print (train.head(10))


# Good! now we have a clean dataset.
# 
# Let's create 3 datasets from it:
# 
# 1. Without Family features
# 
# 2. With LargeFamily feature
# 
# 3. With FamilyClass feature

# In[ ]:


train1 = train.drop(['LargeFamily', 'FamilyClass'], axis=1)
test1  =  test.drop(['LargeFamily', 'FamilyClass'], axis=1)

train2 = train.drop(['FamilyClass'], axis=1)
test2  =  test.drop(['FamilyClass'], axis=1)

train3 = train.drop(['LargeFamily'], axis=1)
test3  =  test.drop(['LargeFamily'], axis=1)

dataset_all = [(train1, test1, 'Without family features'), (train2, test2, 'With large family feature'), (train3, test3, 'With family class feature')]


# Now let's find which classifier works better on each dataset. 

# # Classifier Comparison #

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
	AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

for (train, test, dataset_name) in dataset_all:
    
    X = train.values[0::, 1::]
    y = train.values[0::, 0]

    log_cols = ["Classifier", "Accuracy"]
    log 	 = pd.DataFrame(columns=log_cols)
    
    acc_dict = {}

    for train_index, test_index in sss.split(X, y):
	    X_train, X_test = X[train_index], X[test_index]
	    y_train, y_test = y[train_index], y[test_index]
	
	    for clf in classifiers:
	    	name = clf.__class__.__name__
	    	clf.fit(X_train, y_train)
	    	train_predictions = clf.predict(X_test)
	    	acc = accuracy_score(y_test, train_predictions)
	    	if name in acc_dict:
	    		acc_dict[name] += acc
	    	else:
	    		acc_dict[name] = acc

    for clf in acc_dict:
    	acc_dict[clf] = acc_dict[clf] / 10.0
    	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    	log = log.append(log_entry)

    print ('Classifier Accuracy - ' + dataset_name)
    print (log)
    print ()
        
    plt.figure()
        
    plt.xlabel('Accuracy')
    plt.title('Classifier Accuracy - ' + dataset_name)

    sns.set_color_codes("muted")
    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# Great!
# 
# We can see that adding LargeFamilies feature improves the scores.
# 
# But these results also validate our previous assumption: the classifier score are worse with FamilyClass feature than with LargeFamilies feature. That means that we don't need to know if someone was travelling alone or with a family to make a better prediction. If we use the "alone" information, we over-specialize the classifiers and make them perform worse.
# 
# Let's try to understand why this happens: people travelling with family members have much greater chances of being a kid or a woman than people travelling alone, and we already know that kids and women have the greatest survival chances. That's why knowing if someone was travelling alone has a great impact on survival rate. However, the information of being "alone" is a redundant information, and it is less informative than knowing if someone was a kid or a woman.

# # Prediction #
# Let's use RandomForest classifier to predict our data. Let's also use the dataset with LargeFamily feature.

# In[ ]:


# Use dataset with LargeFamily feature
train, test = train2.values, test2.values
# Use candidate classifier
candidate_classifier = RandomForestClassifier()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
result = candidate_classifier.predict(test)

