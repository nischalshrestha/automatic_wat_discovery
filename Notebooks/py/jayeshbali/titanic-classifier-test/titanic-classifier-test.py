#!/usr/bin/env python
# coding: utf-8

# ## Introduction ##
# 
# This is my first work of machine learning. the notebook is written in python and has inspired from ["Exploring Survival on Titanic" by Megan Risdal, a Kernel in R on Kaggle][1].
# 
# 
#   [1]: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

# In[ ]:


#%matplotlib inline
import numpy as np
import pandas as pd
import re as re
import seaborn as sns


train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})

'''
Append train and test data so that all the data manipulations are common to both
'''
full_data = train.append(test)
full_data.reset_index(inplace=True)

print (train.info())
pd.set_option('display.expand_frame_repr', False)
print (train.describe())
pd.set_option('display.expand_frame_repr', True)
print (train.head())
print (train.describe(include=['O']))
print (train.head())


# # Exploratory Analysis #

# ## 1. Pclass ##
# there is no missing value on this feature and already a numerical value. so let's check it's impact on our train set.

# In[ ]:


print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())


# ## 2. Sex ##

# In[ ]:


print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())


# ## 3. SibSp and Parch ##
# With the number of siblings/spouse and the number of children/parents we can create new feature called Family Size.

# In[ ]:


print (train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean())
print (train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean())
full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
print (full_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# it seems has a good effect on our prediction but let's go further and categorize people to check whether they are alone in this ship or not.

# In[ ]:


full_data['IsAlone'] = 0
full_data.loc[full_data['FamilySize'] == 1, 'IsAlone'] = 1
print (full_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# good! the impact is considerable.

# ## 4. Embarked ##
# the embarked feature has some missing value. and we try to fill those with the most occurred value ( 'S' ).

# In[ ]:


print (full_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).count())
full_data['Embarked'] = full_data['Embarked'].fillna('S')
print (full_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# ## 5. Fare ##
# Fare also has some missing value and we will replace it with the median. then we categorize it into 4 ranges.

# In[ ]:


full_data['Fare'] = full_data['Fare'].fillna(full_data['Fare'].median())
full_data['CategoricalFare'] = pd.qcut(full_data['Fare'], 4)
print (full_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# ## 6. Age ##
# we have plenty of missing values in this feature. # generate random numbers between (mean - std) and (mean + std).
# then we categorize age into 5 range.

# In[ ]:


age_avg 	   = full_data['Age'].mean()
age_std 	   = full_data['Age'].std()
age_null_count = full_data['Age'].isnull().sum()
    
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
full_data['Age'][np.isnan(full_data['Age'])] = age_null_random_list
full_data['Age'] = full_data['Age'].astype(int)
    
full_data['CategoricalAge'] = pd.cut(full_data['Age'], 5)

print (full_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# ## 7. Name ##
# inside this feature we can find the title of people.

# In[ ]:


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

full_data['Title'] = full_data['Name'].apply(get_title)

print(pd.crosstab(full_data['Title'], full_data['Sex']))


#  so we have titles. let's categorize it and check the title impact on survival rate.

# In[ ]:


full_data['Title'] = full_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

full_data['Title'] = full_data['Title'].replace('Mlle', 'Miss')
full_data['Title'] = full_data['Title'].replace('Ms', 'Miss')
full_data['Title'] = full_data['Title'].replace('Mme', 'Mrs')

print (full_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# ## 8. Deck ## 
# A cabin number looks like ‘C123’. The letter refers to the deck, and so we’re going to extract these just like the titles. Let's check the impact of this on the survival rate

# In[ ]:


#Turning cabin number into Deck
cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
#full_data['Deck']=full_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
a= full_data['Cabin'].astype(str).str[0]
full_data['Cabin']=a.str.upper()
print (full_data[['Cabin','Pclass','Survived']].groupby(['Cabin','Pclass'], as_index=False).mean())




# In[ ]:


import matplotlib.pyplot as plt
full_data.columns,train.columns,train.index
#full_data.loc[train.index,:]
f, ax = plt.subplots(figsize=[10,10])
sns.heatmap(full_data.loc[train.index,:].corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Percentage %'},cmap="plasma",ax=ax)
ax.set_title("Correlation Plot")
plt.show()


# # Data Cleaning #
# great! now let's clean our data and map our features into numerical values.

# In[ ]:


full_data['Sex'] = full_data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
# Mapping titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
full_data['Title'] = full_data['Title'].map(title_mapping)
full_data['Title'] = full_data['Title'].fillna(0)
    
    # Mapping Embarked
full_data['Embarked'] = full_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
full_data.loc[ full_data['Fare'] <= 7.91, 'Fare'] 						        = 0
full_data.loc[(full_data['Fare'] > 7.91) & (full_data['Fare'] <= 14.454), 'Fare'] = 1
full_data.loc[(full_data['Fare'] > 14.454) & (full_data['Fare'] <= 31), 'Fare']   = 2
full_data.loc[ full_data['Fare'] > 31, 'Fare'] 							        = 3
full_data['Fare'] = full_data['Fare'].astype(int)
    
    # Mapping Age
full_data.loc[ full_data['Age'] <= 16, 'Age'] 					         = 0
full_data.loc[(full_data['Age'] > 16) & (full_data['Age'] <= 32), 'Age'] = 1
full_data.loc[(full_data['Age'] > 32) & (full_data['Age'] <= 48), 'Age'] = 2
full_data.loc[(full_data['Age'] > 48) & (full_data['Age'] <= 64), 'Age'] = 3
full_data.loc[ full_data['Age'] > 64, 'Age']                             = 4

# Mapping Cabin
cabin_mapping={"A": 1, "B": 2, "C": 3, "D": 4, "E": 5,"F": 6,"G": 7 , "T":8,"N":0}
full_data['Cabin'] = full_data['Cabin'].map(cabin_mapping)
full_data['Cabin'] = full_data['Cabin'].fillna(0)

full_data.head()

# Feature Selection
drop_elements = ['index','Name', 'Ticket', 'SibSp',                 'Parch', 'FamilySize']
full_data = full_data.drop(drop_elements, axis = 1)
full_data = full_data.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)






# In[ ]:


print (full_data[full_data['Survived'].isnull()])


# good! now we have a clean dataset and ready to predict. let's find which classifier works better on this dataset. 

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

print("Start classifer comparison")

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

log_cols = ["Classifier", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)


train = full_data.iloc[:891]
test = full_data.iloc[891:]
targets = full_data['Survived'].iloc[:891]

test.drop('Survived',axis=1,inplace=True)
train.drop('Survived',axis=1,inplace=True)



#targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values



clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
'''
clf = DecisionTreeClassifier(max_features ='sqrt',splitter='random',max_depth = 50 )
clf = clf.fit(train, targets)
'''
train_predictions = clf.predict(test).astype(int)

df_output = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = train_predictions
df_output[['PassengerId','Survived']].to_csv('titanic_submission_final.csv', index=False)
print("File saved")


X = train
y = targets
acc_dict = {}

for train_index, test_index in sss.split(X, y):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	  
	for clf in classifiers:
		name = clf.__class__.__name__
		clf.fit(train, targets)
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

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


clf = AdaBoostClassifier(n_estimators=60)
clf.fit(train, targets)
result = clf.predict(test).astype(int)

df_output = pd.DataFrame()
aux = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = result
df_output[['PassengerId','Survived']].to_csv('titanic_submission_final_2.csv', index=False)
print("File saved")

