#!/usr/bin/env python
# coding: utf-8

# ## <center> Titanic - Data Preprocessing and Visualization </center>

# - This kernel provides insights of the Titanic data.
# - At the end of this notebook, you will be ready with preprocessed data. You can concentrate more on modelling after using this kernel.
# - NOTE: There is a lot of room for improvement and can try many things.
# - Let's get started.

# ### Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# ### Path to the dataset

# In[ ]:


PATH = '../input/'


# ### Load data

# In[ ]:


train_data = pd.read_csv(PATH + 'train.csv')
test_data = pd.read_csv(PATH + 'test.csv')
gender_submission = pd.read_csv(PATH + 'gender_submission.csv')


# ### Train data

# In[ ]:


train_data.head()


# ### Test data

# In[ ]:


test_data.head()


# ## <center> Visualize and preprocess train data </center>

# ### Describe data

# In[ ]:


train_data.describe()


# ### Columns

# In[ ]:


train_data.columns


# ### Data type of each column

# In[ ]:


train_data.dtypes


# ### Number of missing values

# In[ ]:


column_names = train_data.columns
for column in column_names:
    print(column + ' - ' + str(train_data[column].isnull().sum()))


# - The columns 'Age' and 'Cabin' contains more null values.

# ### Insights
# - 'Survived' is the target column/variable.
# - 'PassengerId', 'Name' and 'Ticket' doesn't contribute to the target variable 'Survived'. So, we can remove it from the data.
# - 'Age' and 'Embarked' has less number of missing value. We have to impute them using different techniques.
# - As there are a lot of missing values in the column 'Cabin', we can remove it from the training data.
# - 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare' doesn't have any missing values. 
# - We can also create new variable like 'total size of the family' from the columns 'SibSp' and 'Parch'.

# ### Visualization of 'Survived' (Target column)
# - As we know, majority of passengers couldn't survive.
# - Data is imbalanced.

# In[ ]:


train_data.Survived.value_counts()


# In[ ]:


plt = train_data.Survived.value_counts().plot('bar')
plt.set_xlabel('Survived or not')
plt.set_ylabel('Passenger Count')


# ### Pclass
# - Majority of them are from 3rd class.

# In[ ]:


plt = train_data.Pclass.value_counts().sort_index().plot('bar', title='')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')


# In[ ]:


train_data[['Pclass', 'Survived']].groupby('Pclass').count()


# In[ ]:


train_data[['Pclass', 'Survived']].groupby('Pclass').sum()


# ### Pclass - Survival probability

# In[ ]:


plt = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot('bar')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')


# - From the above results, we can say that, 1st class has high chance of surviving than the other two classes.

# ### Sex
# - Majority of them are Male.

# In[ ]:


plt = train_data.Sex.value_counts().sort_index().plot('bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Passenger count')


# ### Sex - Survival probability
# - As we see, the survival probaility for Female is more. They might have given more priority to female than male.

# In[ ]:


plt = train_data[['Sex', 'Survived']].groupby('Sex').mean().Survived.plot('bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Survival Probability')


# ### Embarked
# - Most of them are from Southampton(S).

# In[ ]:


plt = train_data.Embarked.value_counts().sort_index().plot('bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Passenger count')


# ### Embarked - Survival probability
# - Survival probability: C > Q > S

# In[ ]:


plt = train_data[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot('bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probability')


# ### SibSp - Siblings/Spouse

# In[ ]:


plt = train_data.SibSp.value_counts().sort_index().plot('bar')
plt.set_xlabel('SibSp')
plt.set_ylabel('Passenger count')


# - As we can see, majority of them have no Siblings/Spouse.

# In[ ]:


plt = train_data[['SibSp', 'Survived']].groupby('SibSp').mean().Survived.plot('bar')
plt.set_xlabel('SibSp')
plt.set_ylabel('Survival Probability')


# - The passengers having one sibling/spouse has more survival probability.
# - '1' > '2' > '0' > '3' > '4'

# ### Parch - Children/Parents

# In[ ]:


plt = train_data.Parch.value_counts().sort_index().plot('bar')
plt.set_xlabel('Parch')
plt.set_ylabel('Passenger count')


# - As we can see, majority of them have no Children/Parents.

# In[ ]:


plt = train_data[['Parch', 'Survived']].groupby('Parch').mean().Survived.plot('bar')
plt.set_xlabel('Parch')
plt.set_ylabel('Survival Probability')


# - The passengers having three children/parents has more survival probability.
# - '3' > '1' > '2' > '0' > '5'

# ### Embarked vs Pclass

# In[ ]:


sns.factorplot('Pclass', col = 'Embarked', data = train_data, kind = 'count')


# ### Pclass vs Sex
# - Majority of the passengers are Male in every class. But, the survival probability for female is high.

# In[ ]:


sns.factorplot('Sex', col = 'Pclass', data = train_data, kind = 'count')


# ### Embarked vs Sex

# In[ ]:


sns.factorplot('Sex', col = 'Embarked', data = train_data, kind = 'count')


# ### Create a new feature 'Family size' from the features 'SibSp' and 'Parch'

# In[ ]:


train_data.head()


# In[ ]:


train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1


# In[ ]:


train_data.head()


# ### Remove unnecessary columns
# - We can remove 'Ticket' and 'PassengerId', as they don't contribute to target class.
# - Remove 'Cabin' as it has a lot of missing values in both train and test data

# In[ ]:


train_data = train_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])


# In[ ]:


train_data.head()


# ### Map 'Sex' and 'Embarked' to numerical values.

# In[ ]:


train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'C':0, 'Q':1, 'S':2})


# In[ ]:


train_data.head()


# ### Preprocess 'Name'
# - Extarct title from name of the passenger and categorize them.
# - Drop the column 'Name'

# In[ ]:


train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data = train_data.drop(columns='Name')


# In[ ]:


train_data.Title.value_counts().plot('bar')


# - Combine some of the classes and group all the rare classes into 'Others'.

# In[ ]:


train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')


# In[ ]:


plt = train_data.Title.value_counts().sort_index().plot('bar')
plt.set_xlabel('Title')
plt.set_ylabel('Passenger count')


# - The passengers with title 'Mr' are more.

# In[ ]:


plt = train_data[['Title', 'Survived']].groupby('Title').mean().Survived.plot('bar')
plt.set_xlabel('Title')
plt.set_ylabel('Survival Probability')


# - The survival probability for 'Mrs' and 'Miss' is high comapred to other classes.

# ### Map 'Title' to numerical values

# In[ ]:


train_data['Title'] = train_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})


# In[ ]:


train_data.head()


# ### Correlation between columns

# In[ ]:


corr_matrix = train_data.corr()


# In[ ]:


import matplotlib.pyplot as plt
plt.figure(figsize=(9, 8))
sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2)


# - There are no very highly correlated columns.

# ### Handling missing values

# In[ ]:


train_data.isnull().sum()


# ### Impute 'Embarked' with it's majority class.

# In[ ]:


train_data['Embarked'].isnull().sum()


# - There are two null values in the column 'Embarked'. Let's impute them using majority class.
# - The majority class is 'S'. Impute the unkonown values (NaN) using 'S'

# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna(2)
train_data.head()


# ### Missing values - 'Age'
# - Let's find the columns that are useful to predict the value of Age.

# In[ ]:


corr_matrix = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].corr()


# In[ ]:


plt.figure(figsize=(7, 6))
sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2)


# - Age is not correlated with 'Sex' and 'Fare'. So, we don't consider these two columns while imputing 'Sex'.
# - 'Pclass', 'SibSp' and 'Parch' are negatively correlated with 'Sex'.
# - Let's fill Age with the median age of similar rows from 'Pclass', 'SibSp' and 'Parch'. If there are no similar rows, fill the age with the median age of total dataset.

# In[ ]:


NaN_indexes = train_data['Age'][train_data['Age'].isnull()].index


# In[ ]:


for i in NaN_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == train_data.iloc[i]["SibSp"]) & (train_data.Parch == train_data.iloc[i]["Parch"]) & (train_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        train_data['Age'].iloc[i] = pred_age
    else:
        train_data['Age'].iloc[i] = train_data['Age'].median()


# In[ ]:


train_data.isnull().sum()


# - There are no missing values in the data.

# In[ ]:


train_data.head()


# ## <center> Preprocess test data </center>

# ### Read test data

# In[ ]:


test_data = pd.read_csv(PATH + 'test.csv')


# In[ ]:


test_data.isnull().sum()


# ### Drop 'Ticket', 'PassengerId' and 'Cabin' columns

# In[ ]:


test_data = test_data.drop(columns=['Ticket', 'PassengerId', 'Cabin'])


# In[ ]:


test_data.head()


# ### Convert 'Sex' and 'Embarked' to Numerical values

# In[ ]:


test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})
test_data['Embarked'] = test_data['Embarked'].map({'C':0, 'Q':1, 'S':2})


# In[ ]:


test_data.head()


# ### Extract 'Title' from 'Name' and convert to Numerical values.

# In[ ]:


test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data = test_data.drop(columns='Name')

test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

test_data['Title'] = test_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})


# In[ ]:


test_data.head()


# ### Number of missing values

# In[ ]:


test_data.isnull().sum()


# ### Impute 'Age' using median of columns 'SibSp', 'Parch' and 'Pclass'

# In[ ]:


NaN_indexes = test_data['Age'][test_data['Age'].isnull()].index

for i in NaN_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == test_data.iloc[i]["SibSp"]) & (train_data.Parch == test_data.iloc[i]["Parch"]) & (test_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        test_data['Age'].iloc[i] = pred_age
    else:
        test_data['Age'].iloc[i] = train_data['Age'].median()


# ### Impute 'Title' with it's mode

# In[ ]:


title_mode = train_data.Title.mode()[0]
test_data.Title = test_data.Title.fillna(title_mode)


# ### Impute 'Fare' with it's mean

# In[ ]:


fare_mean = train_data.Fare.mean()
test_data.Fare = test_data.Fare.fillna(fare_mean)


# ### Create a new feature 'FamilySize' from 'SibSp' and 'Parch'

# In[ ]:


test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1


# In[ ]:


test_data.head()


# ### Split 'train data' into 'training data' and 'validation data'

# In[ ]:


train_data.head()


# In[ ]:


from sklearn.utils import shuffle
train_data = shuffle(train_data)


# In[ ]:


# training_data, valid_data = train_test_split(train_data, test_size=0.2)


# In[ ]:


X_train = train_data.drop(columns='Survived')
y_train = train_data.Survived
y_train = pd.DataFrame({'Survived':y_train.values})


# In[ ]:


# X_valid = valid_data.drop(columns='Survived')
# y_valid = valid_data.Survived


# In[ ]:


X_test = test_data


# ## <center> Preprocessed data </center>

# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


X_test.head()


# - It's time to use this preprocessed data and apply different modelling algorithms.
# - Hope this kernel helps you.
# - Don't forget to UPVOTE, if you find this kernel interesting.

# ### Save data

# In[ ]:


X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)

# X_valid.to_csv('X_valid.csv', index=False)
# y_valid.to_csv('y_valid.csv', index=False)

X_test.to_csv('X_test.csv', index=False)

