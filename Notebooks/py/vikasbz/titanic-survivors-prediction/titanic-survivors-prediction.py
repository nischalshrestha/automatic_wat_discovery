#!/usr/bin/env python
# coding: utf-8

# **Import necessary libraries**
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import tree, preprocessing

#%matplotlib inline
#sbn.set()


# **Read Data.**

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# Notice the missing values in Age, Cabin and Embarked.

# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


test.info()


# Notice the missing values in Age, Fare and Cabin.
# 
# Now, for the sake of cleaning the data, we will combine the two datasets, either impute the missing values in the variables, or drop them. Next we will transform few variables to suit our needs. So here is the agenda:
# 1. Missing value analysis.
# 2. Feature engineering

# First, keep the 'Survived' variable from the train data for safe-keeping. And combine the two datasets for missing value analsis and feature engineering.

# In[ ]:


Survived = train['Survived']
Survived.is_copy = False

train.drop('Survived', axis=1, inplace=True)
print('All Good!')


# In[ ]:


data = pd.concat([train, test])

data.describe()


# In[ ]:


data.info()


# Notice the missing values in Age, Fare, Cabin and Embarked.
# We will now take care of these one by one.
# 
# Age:

# In[ ]:


print(data['Age'].isnull().sum())

round( sum(pd.isnull(data['Age'])) / len(data['PassengerId']) * 100, 2)


# Around 20% of the 'Age' values are missing. Let's visualize data distribution for Age: 

# In[ ]:


ax = data["Age"].hist(color='teal', alpha=0.6)
ax.set(xlabel='Age', ylabel='Count')
plt.show()


# Age distribution is right-skewed. So let's impute the missing values with the median.
# 
# Fare:
# Since only one value is missing, we will impute it with the mean.
# 
# Cabin:

# In[ ]:


print(data['Cabin'].isnull().sum())

round(sum(pd.isnull(data['Cabin'])) / len(data['PassengerId']) * 100, 2)


# A whooping 77% of the values are missing. We will definitely drop this varaibale!
# 
# Embarked:

# In[ ]:


print(data['Embarked'].isnull().sum())

round(sum(pd.isnull(data['Embarked'])) / len(data['PassengerId']) * 100, 2)


# Only two values are missing. Let's visualize Embarked:

# In[ ]:


sbn.countplot(x='Embarked', data=data, palette='Set2')
plt.show()


# We will impute Embarked missing values with 'S' - the most frequent value (mode)!

# So the missing value analysis:
# 1. Impute Age NAs with the median.
# 2. Impute Fare NAs with the mean.
# 3. Drop Cabin variable.
# 4. Impute Embarked NAs with 'S'.
# 
# Let's apply these:

# In[ ]:


data['Age'].fillna(data['Age'].median(skipna=True), inplace=True)

data['Fare'].fillna(data['Fare'].mean(skipna=True), inplace=True)

data.drop('Cabin', axis=1, inplace=True)

data['Embarked'].fillna('S', inplace=True)

print('All Good!')


# Let's check the data now:

# In[ ]:


data.info()


# No more missing values. Great work!
# 
# Now let's explore other variables: Sex, SibSp, Parch, Name, Ticket.

# **Feature Engineering**

# In[ ]:


data.head()


# Feature engineering decisions, based on observations:
# 1. Drop Name and Ticket variables, as these are not very useful for us!
# 2. Transform SibSp and Parch into one categorical variable.
# 3. Convert Pclass to categorical variable.
# 4. Convert Sex to categorical variable.
# 5. Convert Embarked to categorical variable. 

# In[ ]:


data.drop('Name', axis=1, inplace=True)

data.drop('Ticket', axis=1, inplace=True)

print('All Good!')


# In[ ]:


data['Family'] = data['SibSp'] + data['Parch']

data['Alone'] = np.where(data['Family'] > 0, 0, 1)

print('All Good!')


# In[ ]:


data.drop('SibSp', axis=1, inplace=True)

data.drop('Parch', axis=1, inplace=True)

data.drop('Family', axis=1, inplace=True)

print('All Good!')


# In[ ]:


data.head()


# In[ ]:


print(data['Pclass'].unique())
print(data['Sex'].unique())
print(data['Embarked'].unique())


# In[ ]:


#Convert Sex into categorical variable:
data = pd.get_dummies(data, columns=['Pclass', 'Sex', 'Embarked'])

data.head()


# Now drop the Sex_male variable which is redundant!

# In[ ]:


data.drop('Sex_male', axis=1, inplace=True)

print('All Good!')


# In[ ]:


data.head()


# One last thing! We need to drop the PassengerId variable too. It has no influence on survivability but it will dominate the model. However, we will keep the original values for safe-keeping and future-use.

# In[ ]:


PassengerId = data['PassengerId']

data.drop('PassengerId', axis=1, inplace=True)

print('All Good!')


# In[ ]:


data['Pclass_1'] = data.Pclass_1.apply(lambda x: int(x))
data['Pclass_2'] = data.Pclass_2.apply(lambda x: int(x))
data['Pclass_3'] = data.Pclass_3.apply(lambda x: int(x))
data['Sex_female'] = data.Sex_female.apply(lambda x: int(x))
data['Embarked_C'] = data.Embarked_C.apply(lambda x: int(x))
data['Embarked_Q'] = data.Embarked_Q.apply(lambda x: int(x))
data['Embarked_S'] = data.Embarked_S.apply(lambda x: int(x))

data.head()

data.info()


# Perfect! Now, we only have the relevant variables. All of them are either numeric, or categorical with binary values.

# **Initial Models**
# 
# Time to test some models based on common sense.

# Before moving ahead, let's split the data back into train and test sets! Here:

# In[ ]:


print('Original datasets')
print(len(train['PassengerId'])) #Original training set
print(len(test['PassengerId'])) #Original testing set

#Add PassengerId back to the dataset
data['PassengerId'] = PassengerId 

train_df = data.iloc[:891] #891 data rows to training set
test_df = data.iloc[891:] #Remaining data rows to the testing set.

train_df.is_copy = False
test_df.is_copy = False

#Verify
print('\nFinal datasets')
print(len(train_df['PassengerId']))
print(len(test_df['PassengerId']))


# Now, add back the Survived (Y_values) variable to train_df dataset!

# In[ ]:


train_df['Survived'] = Survived

train_df.head()


# Now, let's explore the dataset and analyse.

# In[ ]:


sbn.countplot(x='Survived', data=train_df)


# **No Survivors**
# 
# We can see that less people survived. Let's explore Null hypothesis scenario:
# Let's assume nobody survived!
# Now, this is a very bad model. Because we know for a fact that people survived! Nonetheless, it's a possibility to explore.
# Let's see.

# In[ ]:


test_df['Survived'] = 0

no_survivors = test_df[['PassengerId', 'Survived']]
no_survivors.is_copy = False

no_survivors.head()


# In[ ]:


no_survivors.to_csv("no_survivors.csv", index=False)


# no_survivors has 62.67% accuracy. That's too bad.
# 
# 
# **Female Survivors**
# 
# Now let's another intuition. All women survived!

# In[ ]:


sbn.factorplot(x='Survived', col='Sex_female', kind='count', palette='Set1', data=train_df);


# More female passengers survived than male passengers! Let's explore more and quantify this.

# In[ ]:


print(train_df.groupby(by=['Sex_female']).Survived.sum())
print(train_df.groupby(by=['Sex_female']).Survived.count())

train_df.groupby(by=['Sex_female']).Survived.sum()/train_df.groupby(by=['Sex_female']).Survived.count()


# 74% of the Female passengers survived while only 18% of the male passengers suvived.
# Let's build a model on this intuition and submit!

# In[ ]:


test_df.drop('Survived', axis=1, inplace=True)

test_df['Survived'] = test_df['Sex_female'] == 1

test_df['Survived'] = test_df.Survived.apply(lambda x: int(x))
test_df.head()


# In[ ]:


female_survivors = test_df[['PassengerId', 'Survived']]
female_survivors.is_copy = False

female_survivors.head()


# In[ ]:


female_survivors.to_csv("female_survivors.csv", index=False)


# **Decision Tree Classifier**
# 
# Now, let's build our first machine learning model using Decision Tree classifier. We will use scikit library for this.

# First prepare the data for the model. 
# 1. Store the 'Survived' variable from train_df outside in Train_survived.
# 2. Drop 'Survived' variable!
# 3. Convert train_df, test_df and Train_survived to arrays.

# In[ ]:


Train_survived = train_df['Survived']
Train_survived.is_copy = False

train_df.drop('Survived', axis=1, inplace=True)

print('All Good!')


# In[ ]:


X = train_df.values
y = Train_survived.values

test_df.drop('Survived', axis=1, inplace=True)
test = test_df.values

print('All Good!')


# Decision Tree with max_depth = 5

# In[ ]:


#From sklearn:
dtc5 = tree.DecisionTreeClassifier(max_depth=5)
dtc5.fit(X, y)


# In[ ]:


Y_pred5 = dtc5.predict(test)

print('All Good!')


# Let's submit Decision Tree with max_depth = 5!

# In[ ]:


test_df['Survived'] = Y_pred5

dtc5_survivors = test_df[['PassengerId', 'Survived']]
dtc5_survivors.is_copy = False

dtc5_survivors.to_csv("dtc5_survivors.csv", index=False)


# Let's check with max_depth = 5

# In[ ]:


dtc3 = tree.DecisionTreeClassifier(max_depth=3)
dtc3.fit(X, y)


# In[ ]:


Y_pred3 = dtc3.predict(test)

print('All Good!')


# In[ ]:


test_df.drop('Survived', axis=1, inplace=True)

test_df['Survived'] = Y_pred3

dtc3_survivors = test_df[['PassengerId', 'Survived']]
dtc3_survivors.is_copy = False

dtc3_survivors.to_csv("dtc3_survivors.csv", index=False)

