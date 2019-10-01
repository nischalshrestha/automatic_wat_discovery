#!/usr/bin/env python
# coding: utf-8

# ## Introduction ##
# 
# This kernel is forked from Sina's elgant work: [Titanic best working Classifier][1] with family/group survival feature extracted from the data. The family/group information is extracted from name, fare and ticket number through close examing of the data and insparation I got from reading the discussions on the competition. I believe it is the first time this feature has been used as a prediction feature (or at least I have not browsed all the kernels to see it being used :-)). This new feature improved the score by ~1.5% and put the score to be 0.81818. I believe this feature can be used in other models to improve the prediction accuracy as well.
# 
#   [1]: https://www.kaggle.com/sinakhorami/titanic-best-working-classifier?scriptVersionId=560373

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import re as re

train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]

print (train.info())


# In[ ]:


print(len(train))


# # Feature Engineering #

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


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# it seems has a good effect on our prediction but let's go further and categorize people to check whether they are alone in this ship or not.

# In[ ]:


for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# good! the impact is considerable.

# ## 4. Embarked ##
# the embarked feature has some missing value. and we try to fill those with the most occurred value ( 'S' ).

# In[ ]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# ## 5. Fare ##
# Fare also has some missing value and we will replace it with the median. then we categorize it into 4 ranges.

# In[ ]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# ## 6. Age ##
# we have plenty of missing values in this feature. # generate random numbers between (mean - std) and (mean + std).
# then we categorize age into 5 range.

# In[ ]:


for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# ## 7. Name ##
# Another way of getting the title

# In[ ]:


for dataset in full_data:
    dataset['Title'] = [x[1].split(".")[0].strip(" ") for x in dataset['Name'].str.split(",")]

print(pd.crosstab(train['Title'], train['Sex']))


#  so we have titles. let's categorize it and check the title impact on survival rate.

# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# # Extracting family information

# First we can use last name to divide the passengers into families. And if you closely examin the data, same family are paying the same fare for the tickets. This suggests the fare is for the family. We can use both last name and fare to grout passengers into families in case different families with the same last name. 

# In[ ]:


train_size = len(train)
test_size = len(test)

all_df = train.append(test)
all_df = all_df[list(train.columns)]

all_df.set_index(['PassengerId'], inplace=True) ## This is to make sure of a unique index for both train & test

## Processing family information
all_df['Last name'] = all_df['Name'].apply(lambda x: str.split(x, ",")[0])
all_df['Fare'].fillna(all_df['Fare'].mean(), inplace=True)

# The Fare is actually for the whole family
fare_df = all_df.loc[all_df['FamilySize']>1, ["Last name", "Fare", "FamilySize"]].iloc[:train_size]
fare_diff = (((fare_df.groupby(['Last name', 'FamilySize']).max() 
 - fare_df.groupby(['Last name', 'FamilySize']).min())!=0).sum()/train_size * 100)
print(("Percentage of families with different fares is: %.1f" %(fare_diff.values[0])) + '%')
# The data shows only 1.7% has a different fare value between family memebers. It's some type of anomaly
# Will use last name and fare to group passengers into families
# First would like to show there is value in doing this
train_temp_df = all_df.iloc[:train_size]
family_df_grpby = train_temp_df[train_temp_df['FamilySize']>1][
    ['Last name', 'Fare', 'FamilySize', 'Survived']].groupby(['Last name', 'Fare'])
family_df = pd.DataFrame(data=family_df_grpby.size(), columns=['Size in train'])
family_df['Survived total'] = family_df_grpby['Survived'].sum().astype(int)
family_df['FamilySize'] = family_df_grpby['FamilySize'].mean().astype(int)
#family_df = family_df[family_df['FamilySize']==8]
print("Whole family survived: %.1f" 
      %(100*len(family_df[family_df['Size in train']==family_df['Survived total'] ])/len(family_df))+'%') 
print("Whole family perished: %.1f" 
      %(100*len(family_df[family_df['Survived total'] == 0])/len(family_df))+'%') 
## Majority family either all perished or all survived, this means we can use this as one feature to 
## predict survival

# Now let's do the feature extraction
# Intialize all 'Family survival', meaning there is no information on if any family members survived. 
# This number can be tuned I guess but I will use it to start with.
grp_partial_age = 0
grp_partial_cabin = 0
grp_age_diff_df = pd.DataFrame()
all_df['Family survival'] = 0.5
for grp, grp_df in all_df[['Survived','Name', 'Last name', 'Fare', 
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last name', 'Fare']):
    if (len(grp_df) != 1):
        grp_missing_age = len(grp_df[grp_df['Age'].isnull()])
        is_partial_age = (grp_missing_age != 0) & (grp_missing_age != len(grp_df))
        grp_partial_age += is_partial_age
        
        sibsp_df = grp_df.loc[grp_df['SibSp']!=0, ['Age']]
        #print(sibsp_df.info())
        sibsp_age_diff = sibsp_df.max() - sibsp_df.min()
        grp_age_diff_df = grp_age_diff_df.append(sibsp_age_diff, ignore_index=True)

        grp_missing_cabin = len(grp_df[grp_df['Cabin'].isnull()])
        grp_partial_cabin += (grp_missing_cabin != 0) & (grp_missing_cabin != len(grp_df))


        for PassID, row in grp_df.iterrows():
            ## Find out if any family memebers survived or not
            smax = grp_df.drop(PassID)['Survived'].max()
            smin = grp_df.drop(PassID)['Survived'].min()

            ## If any family memebers survived, put this feature as 1
            if (smax==1.0): all_df.loc[PassID, 'Family survival'] = 1
            ## Otherwise if any family memebers perished, put this feature as 0
            elif (smin==0.0): all_df.loc[PassID, 'Family survival'] = 0

print("Number of passenger with family survival information: " 
      +str(all_df[all_df['Family survival']!=0.5].shape[0]))

print('partial age group: ' + str(grp_partial_age))
print('partial cabin group: ' + str(grp_partial_cabin))
print(grp_age_diff_df.describe())


# # Extracting group information

# In addtional to family, if you examin the data closely, you will see there are groups of people with same ticket number, and they pay the same fare. This suggests group of friends are travelling together. One will think these friends will help each other and will survive or perish at the same time. We will explore this informtion here.

# In[ ]:


# First find out how many such groups exists that are not families and what is the chance of 
# passengers within the same group survive or perish together
train_temp_df = all_df.iloc[:train_size]
ticket_grpby = train_temp_df.groupby('Ticket')
ticket_df = pd.DataFrame(data=ticket_grpby.size(), columns=['Size in train'])
ticket_df['Survived total'] = ticket_grpby['Survived'].sum().astype(int)
ticket_df['Not family'] = ticket_grpby['Last name'].unique().apply(len)
#ticket_df['Pclass'] = ticket_grpby['Pclass'].median()
ticket_df = ticket_df[(ticket_df['Size in train'] > 1) & (ticket_df['Not family']>1)]
print('Number of groups in training set that is not family: '+ str(len(ticket_df)))
#print("Groups in Pclass 2/3: " + str(len(ticket_df[ticket_df['Pclass']!=1])))
print(("Whole group perished: %.1f" %(100/len(ticket_df)*len(ticket_df[ticket_df['Survived total']==0]))) + '%')
print(("Whole group survived: %.1f" 
       %(100/len(ticket_df)*len(ticket_df[ticket_df['Survived total']==ticket_df['Size in train']]))) + '%')

## Looking at the output, one can see ~76% of group members stay together. So let's extract this feature.
## We will overload the 'Family survival' column instead of creating a seperate feature.
grp_partial_age = 0
grp_partial_cabin = 0
grp_age_diff_df = pd.DataFrame(columns=['Age diff'])
ticket_grpby = all_df.groupby('Ticket')
for _, grp_df in ticket_grpby:
    if (len(grp_df) > 1):
        grp_missing_age = len(grp_df[grp_df['Age'].isnull()])
        grp_partial_age += (grp_missing_age != 0) & (grp_missing_age != len(grp_df))

        grp_age_diff_df = grp_age_diff_df.append(pd.DataFrame(data=[grp_df['Age'].max() 
                                                                    - grp_df['Age'].min()]
                                                              , columns=['Age diff']))


        grp_missing_cabin = len(grp_df[grp_df['Cabin'].isnull()])
        grp_partial_cabin += (grp_missing_cabin != 0) & (grp_missing_cabin != len(grp_df))
        for PassID, row in grp_df.iterrows():
            if (row['Family survival']==0)|(row['Family survival']==0.5):
                smax = grp_df.drop(PassID)['Survived'].max()
                smin = grp_df.drop(PassID)['Survived'].min()
                if (smax==1.0): all_df.loc[PassID, 'Family survival'] = 1
                elif (smin==0.0): all_df.loc[PassID, 'Family survival'] = 0
print('partial age group: ' + str(grp_partial_age))
print('partial cabin group: ' + str(grp_partial_cabin))
print("Number of passenger with family/group survival information: " 
      +str(all_df[all_df['Family survival']!=0.5].shape[0]))
train['Family survival'] = (all_df.iloc[:train_size]['Family survival'].values).astype(float)
test['Family survival'] = (all_df.iloc[train_size:]['Family survival'].values).astype(float)
print(grp_age_diff_df.describe())


# Good, we can see 546 passengers have a family/group survival information. That's a sizable chunk out of the total numbers of passengers. Hopefully it will improve our prediction accuracy

# # Data Cleaning #
# great! now let's clean our data and map our features into numerical values.

# In[ ]:


for dataset in full_data:
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
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test  = test.drop(drop_elements, axis = 1)

print (train.head(10))

train = train.values
test  = test.values


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

X = train[0::, 1::]
y = train[0::, 0]

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

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")


# # Prediction #
# After adding this new feature, it looks like GradientBoostingClassifier or LogisticRegression are better. Nonetheless, we will keep using SVC to see the impact of this new feature.

# In[ ]:


candidate_classifier = SVC()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
result = candidate_classifier.predict(test)
result_df = pd.DataFrame(columns=['PassengerId', 'Survived'], 
                         data=np.array([range(892, 1310), result]).T.astype(int))
result_df.to_csv("prediction.csv", index=False)


# In[ ]:




