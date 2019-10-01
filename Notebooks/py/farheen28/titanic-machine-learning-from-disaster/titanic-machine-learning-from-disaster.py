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


# Loading  data from file.

# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# Getting familiar with the data. Identifying features, their type, shape and size.

# In[ ]:


print("(Rows, Columns):", train.shape)


# In[ ]:


train.head()


# In[ ]:


train.columns


# In[ ]:


train["Survived"].value_counts()


# **Feature Engineering.**

# The survival rate is around 38% overall for any passenger who boards the ship. But we would like to know on what features the survival is dependent so we can more  accurately predict the survival. Thus let's see, how each feature affects survival rate.

# In[ ]:


train.info()


# Categorical Variables:
# nominal: Survived, Sex, and Embarked. 
# Ordinal: Pclass.
# 
# Numerical
# Continous: Age, Fare. 
# Discrete: SibSp, Parch.
# 
# String:
# Name
# Ticket, Cabin(alphanumeric)
# 
# Target(Dependent): survived
# Independent: sex, embarked, Pclass, age, Fare, SibSp, Parch, Name, Ticket, Cabin

# 1. Pclass vs survival
# 2. Name vs survival
# 3. Sex vs survival
# 4. Age vs survival
# 5. Alone not alone vs  vs survival
# 6. Fare vs survival
# 7. Cabin vs survival
# 8. Embarked vs survival

# In[ ]:


train["Pclass"].value_counts()


# In[ ]:


train["Sex"].value_counts()


# In[ ]:


train["Embarked"].value_counts()


# Let's see now, how survival depends on each feature.

# In[ ]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Now, it may seem that the 'Cabin' feature don't really affect the survival rate. Now survival rate will be more for people resting in cabins which were not hit by iceberg and were not affected. Moreover, since one cabin is shared by more than one person, the survival of a person would be greater belonging to the cabin, having higher rate of survival. I choose to keep the cabin feature.

# **Wrangle Data**

# There are some features which better correlates to survival rate on being combined or being extracted from existing features. That's what we will be doing.
# Extract Feature: Name(We'll be extracting the title from this feature).
# Group feature: Age, Fare.
# Combine Feature: sibSpl and Parch into Family size.

# But before we start prepocessing the data, we would better drop some features, to speed up our analysis. But first, we have to decide the correlation among features and the survival rate, as a parameter to decide which features to drop.

# In[ ]:


#Overall Correlation
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(train.corr(),annot=True)
plt.show()


# **Data Cleaning**

# In[ ]:


for col in train:
    val=train[col].isnull().sum()
    if val>0.0:
        print("Number of missing values in column ",col,":",val)


# From the table, we can drop some features which donot contribute to the output like passengerId and ticket.

# In[ ]:


combine = [train, test]


# In[ ]:


print("Before", train.shape, test.shape, combine[0].shape, combine[1].shape)

train = train.drop(['Ticket','PassengerId'], axis=1)
test = test.drop(['Ticket'], axis=1)
combine = [train, test]

print("After", train.shape, test.shape, combine[0].shape, combine[1].shape)


# Now, we extract the titles from name and replace name with the titles and see if it is correlated to survival rate.

# In[ ]:


for dataset in combine:
    dataset['Name'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Name'], train['Sex'])


# In[ ]:


for dataset in combine:
    dataset['Name'] = dataset['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Name'] = dataset['Name'].replace('Mlle', 'Miss')
    dataset['Name'] = dataset['Name'].replace('Ms', 'Miss')
    dataset['Name'] = dataset['Name'].replace('Mme', 'Mrs')
    
train[['Name', 'Survived']].groupby(['Name'], as_index=False).mean()


# As we had predicted, the titles do correlate with the survival rate.
# Let's now map all catagorical variables into integer type for better analysis

# In[ ]:


name_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for dataset in combine:
    dataset['Name'] = dataset['Name'].map(name_mapping)
    dataset['Name'] = dataset['Name'].fillna(0)

train.head()


# In[ ]:


#mapping sex.
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()


# There are only two missing values in embarkment, and we will fill them with the most recurring value in the datset.

# In[ ]:


freq_port = train.Embarked.dropna().mode()[0]
freq_port 


# In[ ]:


#mapping embarkment.
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port) 
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train.head()


# Now, we create new features by combining number of sibling and number of parents into a family and see how the surviavl rate depends on wheather a passenger is boarding the ship alone or with family.

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# In[ ]:


train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# From the above table se we that the family size 2,3,4 have higher survival rate while family size greater than 4 and passengers boarding alone has a lower survival rate.  We can fashion a new feature mapping the family size. The correlation is strong and thus this feature should be kept.

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['FamilySize'].map( {1: 0,2:1,3:1,4:1,5:0,6:0,7:0,8:0,11:0 } ).astype(int)
train.head()


# For continuous features like age and fare, our analysis would be much easier if we divided into ranges.

# In[ ]:


guess_ages=np.zeros((2,3))
guess_ages


# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train.head()


# We now create age groups and replace them with ordinals for easier classification.

# In[ ]:


train['AgeGroup'] = pd.cut(train['Age'], 5)
train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup', ascending=True)


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train.head()


# Age and Pclass have a strong correlation among then, so we may as well, combine them into a single feature Age*Pclass.

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[ ]:


Now, we discretize fare.


# In[ ]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()


# In[ ]:


#fare ranges
train['FareRange'] = pd.qcut(train['Fare'], 4)
train[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='FareRange', ascending=True)


# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train.head(10)


# In[ ]:


train = train.drop(['Pclass', 'Age', 'SibSp', 'Parch','FareRange','Cabin','AgeGroup'], axis=1)
test = test.drop(['Pclass', 'Age', 'SibSp', 'Parch','Cabin'], axis=1)


# In[ ]:


train.head()


# In[ ]:


test.head()


# **Building Model and Prediction.**
# This is a case of Supervised and Classfication and Regression Problem.
# The first model to test would be **Logistic Regression**.

# In[ ]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
#submission.to_csv('../output/gender_submission.csv', index=False)


# **References**
# A journey through Titanic
# Getting Started with Pandas: Kaggle's Titanic Competition
# [https://www.kaggle.com/startupsci/titanic-data-science-solutions]
# [https://www.kaggle.com/sinakhorami/titanic-best-working-classifier]
# [https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic]
# [https://www.kaggle.com/nicapotato/titanic-voting-pipeline-stack-and-guide]
# [https://www.kaggle.com/sanjaydeo96/titanic-data-analysis-and-ml]
# [https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python]
