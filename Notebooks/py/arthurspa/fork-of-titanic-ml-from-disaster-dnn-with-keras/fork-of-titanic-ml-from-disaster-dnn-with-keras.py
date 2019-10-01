#!/usr/bin/env python
# coding: utf-8

# In[54]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Allows showing more than one DataFrame in the same jupyter cell
# Solution found in https://stackoverflow.com/questions/34398054/ipython-notebook-cell-multiple-outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Data visualization
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir('../input'))

# Any results you write to the current directory are saved as output.


# ## Acquire data

# In[55]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
combine = [train_data, test_data]


# # Analyzing dataset

# ## Input features

# In[56]:


print(train_data.columns.values)


# ## Visualize some train samples

# In[57]:


train_data.head()
train_data.info()

test_data.tail()
test_data.info()


# We have 1 missing Fare value in test_data that we'll fix later.

# ## Distribuition of features

# ### Numeric Features

# In[58]:


train_data.describe()


# ### Alphanumeric Features

# In[59]:


train_data.describe(include=['O'])


# We have 2 missing Embarked values that we'll fix later.

# # Correlating data

# In[60]:


train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[61]:


train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[62]:


train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[63]:


train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# # Dropping Irrelevant Features

# In[64]:


print(f'Before: number of columns = {len(train_data.columns.values)}')
train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_data, test_data]
print(f'After: number of columns = {len(train_data.columns.values)}')


# # Creating New Features

# ## Title Feature
# 
# Create Title feature from all existing title names
# 

# In[65]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_data['Title'], train_data['Sex'])


# Replace some title names

# In[66]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',                                                  'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],                                                'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# ## Converting categorial features

# Convert Title words to values from 0 to 5

# In[67]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# Now it's safe to drop Name and PassengerId features

# In[68]:


print(f'Before: number of columns = {len(train_data.columns.values)}')
train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Name'], axis=1)
combine = [train_data, test_data]
print(f'After: number of columns = {len(train_data.columns.values)}')


# Map Sex feature to 0 and 1

# In[69]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_data.head()


# # Completing Features
# 
# ## Estimating missing Age feature
# 
# Age will be estimated based on the mean of sets comprising of combinations of Pclass and Gender features.
# 
# Initializing guessed values for Pclass x Gender combinations:
# 

# In[70]:


guess_ages = np.zeros((2,3))
guess_ages


# Iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.

# In[71]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_data = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_data.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int(age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_data.head()


# ## Creating temporarily AgeBand feature
# 
# Create new feature AgeBand ad check its correlation with Survived. This new temporarily feature will be used to convert ages from continuos to discrete values based on the corresponding age bands.

# In[72]:


train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# Now replace Age with ordinals based on these bands.

# In[73]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_data.head()


# Now remove AgeBands feature.

# In[74]:


train_data = train_data.drop(['AgeBand'], axis=1)
combine = [train_data, test_data]
train_data.head()


# ## Create FamilySize based on Parch and SibSp

# In[75]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### Create feature IsAlone base on FamilySize

# In[76]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# Drop Parch, SibSp, and FamilySize features in favor of IsAlone.

# In[77]:


train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_data, test_data]

train_data.head()


# ## Create an artificial feature combining Pclass and Age

# In[78]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# ## Complete missing categorical feature
# 
# Complete 2 missing values from Embarked feature by replacing it with the most frequent port.
# 

# In[79]:


freq_port = train_data.Embarked.dropna().mode()[0]
freq_port


# In[80]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Convert categorical value to int value.

# In[81]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_data.head()


# ## Fixing missing Fare value and Creating FarBand feature
# 
# Replacing the sigle Fare missing value in test_data with the mode (most common Fare value)

# In[82]:


test_data['Fare'].fillna(test_data['Fare'].dropna().mode()[0], inplace=True)
test_data.head()


# Create new feature FareBand

# In[83]:


train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# Convert the Fare feature to ordinal values based on the FareBand

# In[84]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# Drop FareBand
train_data = train_data.drop(['FareBand'], axis=1)
combine = [train_data, test_data]
    
train_data.head(10)


# Final test dataset:

# In[85]:


test_data.head(10)


# # Preparing Datasets
# 

# In[151]:


# Clean train_data
X = train_data.drop('Survived', axis=1)
Y = train_data[['Survived']]
P = test_data.drop('PassengerId', axis=1).copy() # data to predict

def train_dev_test_split(df, train_percent=.7, validate_percent=.1, seed=1):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

d_train, d_dev, d_test = train_dev_test_split(train_data, train_percent=.8, validate_percent=.1)

# train
X_train = d_train.drop('Survived', axis=1)
Y_train = d_train[['Survived']]
print(f'X_train = {X_train.shape}')
print(f'Y_train = {Y_train.shape}')

# dev validation
X_dev = d_dev.drop('Survived', axis=1)
Y_dev = d_dev[['Survived']]
print(f'X_dev = {X_dev.shape}')
print(f'Y_dev = {Y_dev.shape}')

# test validation
X_test = d_test.drop('Survived', axis=1)
Y_test = d_test[['Survived']]
print(f'X_test = {X_test.shape}')
print(f'Y_test = {Y_test.shape}')

# final data to predict and submit
X_predict = test_data.drop('PassengerId', axis=1).copy() # data to predict
print(f'X_predict = {X_predict.shape}')


# ## Building and Train the Model

# In[153]:


# Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#from keras.optimizers import Adam
#from keras.losses import binary_crossentropy

number_of_features = X_train.shape[1]
model = Sequential()
model.add(Dense(60, input_dim=number_of_features, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(.6))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['binary_accuracy'])

model.fit(X_train, Y_train, validation_data=(X_dev,Y_dev), batch_size=32, epochs=100, verbose=0)
scores = model.evaluate(X_test,Y_test, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))


# ## Predict

# In[163]:


predictions = model.predict(X_predict)
predictions = (predictions > .7).astype(int)
predictions[:10]


# ## Prepare to Submit

# In[165]:


submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': predictions.transpose().reshape(test_data['PassengerId'].shape)
    })
submission.head(10)


# ## Submit answer

# In[166]:


submission.to_csv('./submission.csv', index=False)


# In[ ]:




