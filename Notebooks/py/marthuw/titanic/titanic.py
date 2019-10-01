#!/usr/bin/env python
# coding: utf-8

# In this notebook I'll be trying to predict what people survived the Titanic sinking and what people didn't. This is my first classification problem. Let's start by gaining some insight in the available data. Luckily there is only one dataset, so we won't have to merge, concat and that kind of stuff. I am very happy about that.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt # plotting
import seaborn as sns # more plotting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train_test = [train, test]


# In[ ]:


train.head()


# In[ ]:


test.head()


# The dataframe has 12 columns, of which we'll be using 10 to predict the Survived column. We won't use PassengerId, because it doesn't hold any useful information for predicting if someone survived the Titanic sinking. Let's check what columns contain missing values and how many values are missing. First I will look at the categorical variables (technically, some are actually numerical, but because they have such few possible values I just treat them as categorical). I will leave the Ticket out for now because just from the head of the dataframe I can see there is no order in it at all. I will change male and female in the Sex column in 1 and 0 respectively.

# In[ ]:


for df in train_test:
    df['Sex'] = df['Sex'].replace({'male': '1', 'female': '0'})
    df['Sex'] = df['Sex'].astype(int)

for df in train_test:
    for col in df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']].columns:
        print("---- %s ---" % col)
        print(df[col].value_counts(dropna = False))

print(train['Survived'].value_counts(dropna = False))


#  While we are at it let's make some histograms of the categorical variables aswell.

# In[ ]:


for df in train_test:
    df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']].hist()


# In[ ]:


train['Survived'].hist()


# When we look at the value counts of the categorical variables we can see a few interesting things.
# Cabin seems to be all over the place with lots of missing values and lots of different cabins. Later I will check if there is some structure to be found in these cabin values.
# Most people don't have a lot of siblings, spouses, parents and children with them, by far most values are 0 and 1 for SibSp and Parch.
# Most people embarked in Southampton. In the embarked column there are two missing values, I don't know how I will deal with these yet.
# About 65% of the passengers were male, and 62% of the passengers died during the sinking, is it a coïncidence these numbers are so close to eachother?
# The 3rd class tickets were the most popular, with 55%. About 25% had a first class ticket and 20% a second class ticket.
# 
# Let's make some histogram of the numerical features (Age and Fare) next. Also I will check for missing values in these columns.

# In[ ]:


for df in train_test:
    df[['Age', 'Fare']].hist()
    print(df[['Age', 'Fare']].info())
    print(df['Age'].describe())
    print(df['Fare'].describe())


# We can see that the Fare column doesn't have any missing values, but surprisingly there are missing values in the age column. Maybe some high-class ladies weren't willing to admit they were getting old?
# Anyway, I don't see anything weird in the distribution and histogram of Age, the distribution is a little bit skewed to the right, and there are less teenagers (8-17 yo) than brats (0-8 yo).
# In the Fare histogram and distribution I do see some extreme outliers however. We'll have to see later if there is an explanation for this. Maybe these people were of very high status and got very luxurious suites etc. for which they payed a high premium.
# 
# Next let's check for correlations between numerical features. 

# In[ ]:


f, ax = plt.subplots(figsize = (20, 20))

corrmat = train.corr()
sns.heatmap(corrmat, annot = True)


# In[ ]:


f, ax = plt.subplots(figsize = (20, 20))

corrmat = test.corr()
sns.heatmap(corrmat, annot = True)


# A lot of the correlations we see in this matrix make sense. There is a strong negative correlation between Fare and Pclass, so people pay more on average for more luxurious classes. There also is a negative correlation between Age and Fare. Younger people tend to have less money so that makes sense aswell. We can also see that people who brought more parents and children with them on average brought more siblings and spouses (let's hope it is only one spouse, but we will never know) aswell. Some correlations seem to change significantly between the train and test datasets.

# In[ ]:


for df in train_test:
    print(df.groupby('Embarked')['Fare'].mean())


# Here we can see people who embarked from Cherbourg on average paid the most by quite a large margin. This difference is even more clear in the test dataset.
# 
# Let's try to clean the columns with unstructured or missing data. I will start with the easiest one: Embarked. This column has only two missing values in the training set and none in the test set.

# In[ ]:


train.loc[train['Embarked'].isnull()]


# Both persons with missing Embarked data seem to be in the same cabin, this could mean something, but since 2 rows are so insignificant I will just drop these.
# 
# Next I will look at the age column. In both train and test this column has quite a few missing values. In the documentation it says xx.5 values are estimated ages and fractions are used for babies less than 1 year old. It doesn't say anything about missing values however. For now I will just fill in -1.0 where age is missing.

# In[ ]:


train = train.loc[train['Embarked'].isnull() == False]

train.loc[train['Age'].isnull(),'hasAge'] = 0
train['hasAge'] = train['hasAge'].fillna(1)
test.loc[test['Age'].isnull(),'hasAge'] = 0
test['hasAge'] = test['hasAge'].fillna(1)

train['Age'] = train['Age'].fillna(-1.0)
test['Age'] = test['Age'].fillna(-1.0)

train.loc[((train['Age']/0.5)%2 == 1) & (train['Age'] >= 1.0), 'hasAge'] = 2
train.loc[((train['Age']/0.5)%2 == 0) & (train['Age'] >= 1.0), 'hasAge'] = 1
train.loc[train['Age'] < 1.0, 'hasAge'] = 1
train.loc[train['Age'] == -1.0, 'hasAge'] = 0

test.loc[((test['Age']/0.5)%2 == 1) & (test['Age'] >= 1.0), 'hasAge'] = 2
test.loc[((test['Age']/0.5)%2 == 0) & (test['Age'] >= 1.0), 'hasAge'] = 1
test.loc[test['Age'] < 1.0, 'hasAge'] = 1
test.loc[test['Age'] == -1.0, 'hasAge'] = 0

print(train.groupby('hasAge')['Survived'].mean())


# In[ ]:


print(train['Embarked'])


# Next I will try to bring some structure in the cabin column. There is a lot of missing data so I will just start by replacing that with Z99 for now. After that I will split the Cabin column in CabinL (letter) and CabinN (number). I will try to organize everything in groups of two, three or four with one letter and one, two or three numbers. If there is one loose letter and a group I will delete the letter. If there is just a letter I will fill in 999 as the number. If there is just a number I will fill in Z as the letter.

# In[ ]:


train.loc[train['Cabin'].isnull(), 'hasCabin1'] = 0
train['hasCabin1'] = train['hasCabin1'].fillna(1)
test.loc[test['Cabin'].isnull(), 'hasCabin1'] = 0
test['hasCabin1'] = test['hasCabin1'].fillna(1)

train['Cabin'] = train['Cabin'].fillna("Z999")
test['Cabin'] = test['Cabin'].fillna("Z999")

train['CabinL'] = train.loc[train['Cabin'].str.len() == 1.0, 'Cabin']
train['CabinN1'] = "999"
train['CabinN2'] = "999"
train['CabinN3'] = "999"
train['CabinN4'] = "999"
test['CabinL'] = test.loc[test['Cabin'].str.len() == 1.0, 'Cabin']
test['CabinN1'] = "999"
test['CabinN2'] = "999"
test['CabinN3'] = "999"
test['CabinN4'] = "999"

train.loc[train['Cabin'].str.len() <= 4.0, 'CabinL'] = train['Cabin'].astype(str).str[0]
train.loc[train['Cabin'].str.len() <= 4.0, 'CabinN1'] = train['Cabin'].astype(str).str[1:]
test.loc[test['Cabin'].str.len() <= 4.0, 'CabinL'] = test['Cabin'].astype(str).str[0]
test.loc[test['Cabin'].str.len() <= 4.0, 'CabinN1'] = test['Cabin'].astype(str).str[1:]

train.loc[train['Cabin'].str.len() == 5.0, 'CabinL'] = train['Cabin'].astype(str).str[2]
train.loc[train['Cabin'].str.len() == 5.0, 'CabinN1'] = train['Cabin'].astype(str).str[3:]
test.loc[test['Cabin'].str.len() == 5.0, 'CabinL'] = test['Cabin'].astype(str).str[2]
test.loc[test['Cabin'].str.len() == 5.0, 'CabinN1'] = test['Cabin'].astype(str).str[3:]

train.loc[train['Cabin'].str.len() == 7.0, 'CabinL'] = train['Cabin'].astype(str).str[0]
train.loc[train['Cabin'].str.len() == 7.0, 'CabinN1'] = train['Cabin'].astype(str).str[1:3]
train.loc[train['Cabin'].str.len() == 7.0, 'CabinN2'] = train['Cabin'].astype(str).str[5:]
test.loc[test['Cabin'].str.len() == 7.0, 'CabinL'] = test['Cabin'].astype(str).str[0]
test.loc[test['Cabin'].str.len() == 7.0, 'CabinN1'] = test['Cabin'].astype(str).str[1:3]
test.loc[test['Cabin'].str.len() == 7.0, 'CabinN2'] = test['Cabin'].astype(str).str[5:]

train.loc[train['Cabin'].str.len() == 11.0, 'CabinL'] = train['Cabin'].astype(str).str[0]
train.loc[train['Cabin'].str.len() == 11.0, 'CabinN1'] = train['Cabin'].astype(str).str[1:3]
train.loc[train['Cabin'].str.len() == 11.0, 'CabinN2'] = train['Cabin'].astype(str).str[5:7]
train.loc[train['Cabin'].str.len() == 11.0, 'CabinN3'] = train['Cabin'].astype(str).str[-2:]
test.loc[test['Cabin'].str.len() == 11.0, 'CabinL'] = test['Cabin'].astype(str).str[0]
test.loc[test['Cabin'].str.len() == 11.0, 'CabinN1'] = test['Cabin'].astype(str).str[1:3]
test.loc[test['Cabin'].str.len() == 11.0, 'CabinN2'] = test['Cabin'].astype(str).str[5:7]
test.loc[test['Cabin'].str.len() == 11.0, 'CabinN3'] = test['Cabin'].astype(str).str[-2:]

train.loc[train['Cabin'].str.len() == 15.0, 'CabinL'] = train['Cabin'].astype(str).str[0]
train.loc[train['Cabin'].str.len() == 15.0, 'CabinN1'] = train['Cabin'].astype(str).str[1:3]
train.loc[train['Cabin'].str.len() == 15.0, 'CabinN2'] = train['Cabin'].astype(str).str[5:7]
train.loc[train['Cabin'].str.len() == 15.0, 'CabinN3'] = train['Cabin'].astype(str).str[9:11]
train.loc[train['Cabin'].str.len() == 15.0, 'CabinN4'] = train['Cabin'].astype(str).str[-2:]
test.loc[test['Cabin'].str.len() == 15.0, 'CabinL'] = test['Cabin'].astype(str).str[0]
test.loc[test['Cabin'].str.len() == 15.0, 'CabinN1'] = test['Cabin'].astype(str).str[1:3]
test.loc[test['Cabin'].str.len() == 15.0, 'CabinN2'] = test['Cabin'].astype(str).str[5:7]
test.loc[test['Cabin'].str.len() == 15.0, 'CabinN3'] = test['Cabin'].astype(str).str[9:11]
test.loc[test['Cabin'].str.len() == 15.0, 'CabinN4'] = test['Cabin'].astype(str).str[-2:]

train.loc[train['Cabin'].str.len() == 1.0, 'CabinN1'] = "999"
test.loc[test['Cabin'].str.len() == 1.0, 'CabinN1'] = "999"

train['CabinN1'] = pd.to_numeric(train['CabinN1'], downcast='integer')
train['CabinN2'] = pd.to_numeric(train['CabinN2'], downcast='integer')
train['CabinN3'] = pd.to_numeric(train['CabinN3'], downcast='integer')
train['CabinN4'] = pd.to_numeric(train['CabinN4'], downcast='integer')
test['CabinN1'] = pd.to_numeric(test['CabinN1'], downcast='integer')
test['CabinN2'] = pd.to_numeric(test['CabinN2'], downcast='integer')
test['CabinN3'] = pd.to_numeric(test['CabinN3'], downcast='integer')
test['CabinN4'] = pd.to_numeric(test['CabinN4'], downcast='integer')

train['CabinN1'] = train['CabinN1'].astype(int)
test['CabinN1'] = train['CabinN1'].astype(int)

train.loc[train['CabinN2'] == 999, 'hasCabin2'] = 0
train['hasCabin2'] = train['hasCabin2'].fillna(1)
train.loc[train['CabinN3'] == 999, 'hasCabin3'] = 0
train['hasCabin3'] = train['hasCabin3'].fillna(1)
train.loc[train['CabinN4'] == 999, 'hasCabin4'] = 0
train['hasCabin4'] = train['hasCabin4'].fillna(1)

test.loc[test['CabinN2'] == 999, 'hasCabin2'] = 0
test['hasCabin2'] = train['hasCabin2'].fillna(1)
test.loc[test['CabinN3'] == 999, 'hasCabin3'] = 0
test['hasCabin3'] = train['hasCabin3'].fillna(1)
test.loc[test['CabinN4'] == 999, 'hasCabin4'] = 0
test['hasCabin4'] = train['hasCabin4'].fillna(1)


# With the Cabin column out of the way I will now clean the Fare column. During the cleaning and engineering with the Cabin column I came up with a hypothesis for the outliers in the Fare column. Whenever there are multiple cabins in a row, the Fare seems to take on extreme values. Normally I would delete these outliers, but since there seems to be an explanation I will leave them there and just delete the single row with a missing value in the test set. I will also round the Fares to 1 decimal.

# In[ ]:


test.loc[test['Fare'].isnull(), 'Fare'] = test['Fare'].mean()

train['Fare'] = round(train['Fare'], 1)
test['Fare'] = round(test['Fare'], 1)

print(train['Fare'])

train


# Let's engineer a new feature that is a combination of Pclass and gender called class_gender. 0 if male and class 1, 1 if female and class 1 etc. etc.

# In[ ]:


train['gender_class'] = 4 #male Pclass 1

train.loc[((train['Sex'] == 0) & (train['Pclass'] == 1)), 'gender_class'] = 1 #female Pclass 1
train.loc[((train['Sex'] == 1) & (train['Pclass'] == 2)), 'gender_class'] = 5 #male Pclass 2
train.loc[((train['Sex'] == 0) & (train['Pclass'] == 2)), 'gender_class'] = 2 #female Pclass 2
train.loc[((train['Sex'] == 1) & (train['Pclass'] == 3)), 'gender_class'] = 6 #male Pclass 3
train.loc[((train['Sex'] == 0) & (train['Pclass'] == 3)), 'gender_class'] = 3 #female Pclass 3

test['gender_class'] = 4

test.loc[((test['Sex'] == 0) & (test['Pclass'] == 1)), 'gender_class'] = 1
test.loc[((test['Sex'] == 1) & (test['Pclass'] == 2)), 'gender_class'] = 5
test.loc[((test['Sex'] == 0) & (test['Pclass'] == 2)), 'gender_class'] = 2
test.loc[((test['Sex'] == 1) & (test['Pclass'] == 3)), 'gender_class'] = 6
test.loc[((test['Sex'] == 0) & (test['Pclass'] == 3)), 'gender_class'] = 3

print(train.groupby('gender_class')['Survived'].mean())


# In[ ]:


train['isAlone'] = 0
train.loc[(train['SibSp'] == 0) & (train['Parch'] == 0),'isAlone'] = 1

test['isAlone'] = 0
test.loc[(test['SibSp'] == 0) & (test['Parch'] == 0),'isAlone'] = 1

print(train.groupby('isAlone')['Survived'].mean())


# In[ ]:


train.loc[(train['Pclass'] == 1) & (train['isAlone'] == 0), 'class_alone'] = 1 #together Pclass 1
train.loc[(train['Pclass'] == 1) & (train['isAlone'] == 1), 'class_alone'] = 3 #alone Pclass 1
train.loc[(train['Pclass'] == 2) & (train['isAlone'] == 0), 'class_alone'] = 2 #together Pclass 2
train.loc[(train['Pclass'] == 2) & (train['isAlone'] == 1), 'class_alone'] = 4 #alone Pclass2
train.loc[(train['Pclass'] == 3) & (train['isAlone'] == 0), 'class_alone'] = 5 #together Pclass 3
train.loc[(train['Pclass'] == 3) & (train['isAlone'] == 1), 'class_alone'] = 6 #alone Pclass 3

test.loc[(test['Pclass'] == 1) & (test['isAlone'] == 0), 'class_alone'] = 1
test.loc[(test['Pclass'] == 1) & (test['isAlone'] == 1), 'class_alone'] = 3
test.loc[(test['Pclass'] == 2) & (test['isAlone'] == 0), 'class_alone'] = 2
test.loc[(test['Pclass'] == 2) & (test['isAlone'] == 1), 'class_alone'] = 4
test.loc[(test['Pclass'] == 3) & (test['isAlone'] == 0), 'class_alone'] = 5
test.loc[(test['Pclass'] == 3) & (test['isAlone'] == 1), 'class_alone'] = 6

print(train.groupby('class_alone')['Survived'].mean())


# In[ ]:


train.loc[(train['Pclass'] == 1) & (train['hasAge'] == 0), 'class_hasage'] = 3 #no age Pclass 1
train.loc[(train['Pclass'] == 1) & (train['hasAge'] == 1), 'class_hasage'] = 1 #has age Pclass 1
train.loc[(train['Pclass'] == 2) & (train['hasAge'] == 0), 'class_hasage'] = 4 #no age Pclass 2
train.loc[(train['Pclass'] == 2) & (train['hasAge'] == 1), 'class_hasage'] = 2 #has age Pclass 2
train.loc[(train['Pclass'] == 3) & (train['hasAge'] == 0), 'class_hasage'] = 5 #no age Pclass 3
train.loc[(train['Pclass'] == 3) & (train['hasAge'] == 1), 'class_hasage'] = 6 #has age Pclass 3

test.loc[(test['Pclass'] == 1) & (test['hasAge'] == 0), 'class_hasage'] = 3
test.loc[(test['Pclass'] == 1) & (test['hasAge'] == 1), 'class_hasage'] = 1
test.loc[(test['Pclass'] == 2) & (test['hasAge'] == 0), 'class_hasage'] = 4
test.loc[(test['Pclass'] == 2) & (test['hasAge'] == 1), 'class_hasage'] = 2
test.loc[(test['Pclass'] == 3) & (test['hasAge'] == 0), 'class_hasage'] = 5
test.loc[(test['Pclass'] == 3) & (test['hasAge'] == 1), 'class_hasage'] = 6

print(train.groupby('class_hasage')['Survived'].mean())


# In[ ]:


train.loc[(train['Sex'] == 0) & (train['isAlone'] == 0), 'gender_alone'] = 2 #female together
train.loc[(train['Sex'] == 0) & (train['isAlone'] == 1), 'gender_alone'] = 1 #female alone
train.loc[(train['Sex'] == 1) & (train['isAlone'] == 0), 'gender_alone'] = 3 #male together
train.loc[(train['Sex'] == 1) & (train['isAlone'] == 1), 'gender_alone'] = 4 #male alone

test.loc[(test['Sex'] == 0) & (test['isAlone'] == 0), 'gender_alone'] = 2
test.loc[(test['Sex'] == 0) & (test['isAlone'] == 1), 'gender_alone'] = 1
test.loc[(test['Sex'] == 1) & (test['isAlone'] == 0), 'gender_alone'] = 3
test.loc[(test['Sex'] == 1) & (test['isAlone'] == 1), 'gender_alone'] = 4

print(train.groupby('gender_alone')['Survived'].mean())


# In[ ]:


train.loc[(train['Pclass'] == 1) & (train['hasCabin1'] == 0), 'class_hascabin'] = 4 #no cabin Pclass 1
train.loc[(train['Pclass'] == 1) & (train['hasCabin1'] == 1), 'class_hascabin'] = 2 #has cabin Pclass 1
train.loc[(train['Pclass'] == 2) & (train['hasCabin1'] == 0), 'class_hascabin'] = 5 #no cabin Pclass 2
train.loc[(train['Pclass'] == 2) & (train['hasCabin1'] == 1), 'class_hascabin'] = 1 #has cabin Pclass 2
train.loc[(train['Pclass'] == 3) & (train['hasCabin1'] == 0), 'class_hascabin'] = 6 #no cabin Pclass 3
train.loc[(train['Pclass'] == 3) & (train['hasCabin1'] == 1), 'class_hascabin'] = 3 #has cabin Pclass 3

test.loc[(test['Pclass'] == 1) & (test['hasCabin1'] == 0), 'class_hascabin'] = 4
test.loc[(test['Pclass'] == 1) & (test['hasCabin1'] == 1), 'class_hascabin'] = 2
test.loc[(test['Pclass'] == 2) & (test['hasCabin1'] == 0), 'class_hascabin'] = 5
test.loc[(test['Pclass'] == 2) & (test['hasCabin1'] == 1), 'class_hascabin'] = 1
test.loc[(test['Pclass'] == 3) & (test['hasCabin1'] == 0), 'class_hascabin'] = 6
test.loc[(test['Pclass'] == 3) & (test['hasCabin1'] == 1), 'class_hascabin'] = 3

print(train.groupby('class_hascabin')['Survived'].mean())


# In[ ]:


train.loc[train['Age'] < 14.0, 'ageCat'] = 1
train.loc[train['Age'] >= 14.0, 'ageCat'] = 4
train.loc[train['Age'] > 24.0, 'ageCat'] = 2
train.loc[train['Age'] > 40.0, 'ageCat'] = 3
train.loc[train['Age'] > 60.0, 'ageCat'] = 6
train.loc[train['Age'] == -1.0, 'ageCat'] = 5

test.loc[test['Age'] < 14.0, 'ageCat'] = 1
test.loc[test['Age'] >= 14.0, 'ageCat'] = 4
test.loc[test['Age'] > 24.0, 'ageCat'] = 2
test.loc[test['Age'] > 40.0, 'ageCat'] = 3
test.loc[test['Age'] > 60.0, 'ageCat'] = 6
test.loc[test['Age'] == -1.0, 'ageCat'] = 5

print(train.groupby('ageCat')['Survived'].mean())


# In[ ]:


train['famSize'] = train['SibSp'] + train['Parch']
test['famSize'] = test['SibSp'] + test['Parch']

print(train.groupby('famSize')['Survived'].mean())


# In[ ]:


train['farePP'] = (train['Fare'] / (train['famSize'] + 1.0)).round(1)
test['farePP'] = (test['Fare'] / (test['famSize'] + 1.0)).round(1)


# In[ ]:


train.loc[(train['farePP'] >= 26.0), 'fareCat'] = 3
train.loc[(train['farePP'] < 26.0), 'fareCat'] = 2
train.loc[(train['farePP'] < 8.6), 'fareCat'] = 1

test.loc[(test['farePP'] >= 26.0), 'fareCat'] = 3
test.loc[(test['farePP'] < 26.0), 'fareCat'] = 2
test.loc[(test['farePP'] < 8.6), 'fareCat'] = 1

print(train.groupby('fareCat')['Survived'].mean())


# In[ ]:


train.loc[train['Embarked'] == "S", 'Embarked'] = 1
train.loc[train['Embarked'] == "C", 'Embarked'] = 3
train.loc[train['Embarked'] == "Q", 'Embarked'] = 2

test.loc[test['Embarked'] == "S", 'Embarked'] = 1
test.loc[test['Embarked'] == "C", 'Embarked'] = 3
test.loc[test['Embarked'] == "Q", 'Embarked'] = 2

print(train.groupby('Embarked')['Survived'].mean())


# In[ ]:


allcab = plt.hist(train.loc[train['CabinN1'] != 999, 'CabinN1'], bins = range(0, 140, 10), label = 'all')

allcab


# In[ ]:


deadcab = plt.hist(train.loc[(train['Survived'] == 0) & (train['CabinN1'] != 999), 'CabinN1'], bins = range(0, 140, 10), label = 'dead')

deadcab


# In[ ]:


livecab = plt.hist(train.loc[(train['Survived'] == 1) & (train['CabinN1'] != 999), 'CabinN1'], bins = range(0, 140, 10), label = 'live')

livecab
deadcab
allcab


# In[ ]:


plt.hist(train.loc[train['CabinN1'] != 999, 'CabinN1'], bins = range(0, 140, 10), label = 'all', alpha = 0.5)
plt.hist(train.loc[(train['Survived'] == 0) & (train['CabinN1'] != 999), 'CabinN1'], bins = range(0, 140, 10), label = 'dead', alpha = 0.5)
plt.hist(train.loc[(train['Survived'] == 1) & (train['CabinN1'] != 999), 'CabinN1'], bins = range(0, 140, 10), label = 'live', alpha = 0.5)
plt.legend(loc='upper right')
plt.show()


# In[ ]:


for column in train.columns:
    print(train[column].value_counts(dropna = False))


# I feel a bit intimidated by the name and ticket columns, so let's just take a shot at makingand fitting a model and see what happens. Let's just use a decisiontree for now, later I will mess around with other algorithms.

# In[ ]:


#classifier = DecisionTreeClassifier(max_depth = 10, random_state = 1)

#train_dummies = pd.get_dummies(pd.DataFrame())

splitter = train[['Sex', 'gender_class', 'Embarked', 'ageCat', 'fareCat', 'famSize', 'hasAge', 'Pclass', 'Survived', 'hasCabin1', 'isAlone', 'class_alone', 'gender_alone', 'class_hascabin']]

#res = cross_val_score(clf, X, y, scoring='accuracy', cv = 5)
train_set, fake_test = train_test_split(splitter, test_size = 0.2, random_state = 1)
train_set_true = train_set['Survived']
train_set = train_set.loc[:, train_set.columns != 'Survived']
fake_test_true = fake_test['Survived']
fake_test = fake_test.loc[:, train_set.columns != 'Survived']

splitter_true = splitter['Survived']
splitter_set = splitter.loc[:, splitter.columns != 'Survived']
#classifier.fit(train_set, train_set_true)

#print(classifier.feature_importances_)


# In[ ]:


models = [DecisionTreeClassifier(random_state = 1), RandomForestClassifier(random_state = 1), AdaBoostClassifier(random_state = 1), GradientBoostingClassifier(random_state = 1)]

for model in models:
    res = cross_val_score(model, splitter.loc[:, splitter.columns != 'Survived'], splitter['Survived'], scoring='accuracy', cv = 5)
    print(res)


# AdaBoostClassifier and GradientBoostingClassifier seem to produce the best results, so I will use those while gridsearching and finetuning the model.

# In[ ]:


#%time adaboostclass = GridSearchCV(AdaBoostClassifier(random_state = 1), {'n_estimators': range(60, 80, 1), 'learning_rate': [0.875, 0.9, 0.925]}, cv = 3, refit = True, scoring = 'roc_auc').fit(train_set, train_set_true)

#print(adaboostclass.best_estimator_)


# In[ ]:


get_ipython().magic(u"time gradboostclass = GridSearchCV(GradientBoostingClassifier(random_state = 1), {'learning_rate': [0.01, 0.1, 0.5, 0.8, 1.0], 'n_estimators': range(30, 40, 1), 'max_depth': [2, 3, 4]}, cv = 3, refit = True, scoring = 'roc_auc').fit(splitter_set, splitter_true)")

print(gradboostclass.best_estimator_)


# In[ ]:


#%time randforclass = GridSearchCV(RandomForestClassifier(random_state = 1), {'n_estimators': range(40, 60, 1), 'max_depth': [5, 6, 7]}, cv = 3, refit = True, scoring = 'roc_auc').fit(train_set, train_set_true)

#print(randforclass.best_estimator_)


# In[ ]:


#%time logreg = GridSearchCV(LogisticRegression(random_state = 1), {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]}, cv = 3, refit = True, scoring = 'roc_auc').fit(train_set, train_set_true)

#print(logreg.best_estimator_)


# In[ ]:


#classifiers = [adaboostclass, gradboostclass, randforclass, logreg]

#for classifier in classifiers:
#    fake_test_pred = classifier.predict(fake_test)
#   print(accuracy_score(fake_test_true, fake_test_pred))


# In[ ]:


fake_test_pred = gradboostclass.predict(fake_test)
print(accuracy_score(fake_test_true, fake_test_pred))


# In[ ]:


test = test[['Sex', 'ageCat', 'Embarked','fareCat', 'famSize', 'Pclass', 'hasAge', 'hasCabin1', 'gender_class', 'isAlone', 'gender_alone', 'class_alone', 'class_hascabin']]


# In[ ]:


test = test.fillna(999)
#test['CabinL_T'] = 0

pred_df = pd.DataFrame({'PassengerId': range(892,1310),'Survived': gradboostclass.predict(test)})

pred_df.to_csv('submission.csv', index=False)


# In[ ]:


test

