#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import os
print(os.listdir("../input"))


# ## Data Exploration

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


train_df.shape


# Columns **Age**, ** Cabin**, and ** Embarked** contain null values.

# In[ ]:


train_df.isna().any()


# In[ ]:


print('Number of null ages: {}'.format(len([age for age in train_df.Age.isna() if age])))
print('Number of null cabins: {}'.format(len([cabin for cabin in train_df.Cabin.isna() if cabin])))
print('Number of null embarked: {}'.format(len([embarked for embarked in train_df.Embarked.isna() if embarked])))


# There are a high number of null **ages** and **cabins**. Cabins may be correlated with fare and ages may be associated with family units. For the two missing **embarked**, I will fill them in with the most common point of embarkation.
# 
# ### Question 1: Can we correlate cabins with fares?

# In[ ]:


train_df['cabin_blocks'] = [cabin[0] if cabin is not np.NaN else cabin for cabin in train_df.Cabin]
train_df.head()


# In[ ]:


cabin_blocks = [cabin for cabin in train_df.cabin_blocks.unique() if cabin is not np.NaN]
print(cabin_blocks)

def find_average_fare_for_cabin_block(cabin_block: str):
    sum_of_fares = sum([row['Fare'] for _, row in train_df.iterrows() if row['cabin_blocks'] == cabin_block])
    number_of_cabins = len([cabin for cabin in train_df.cabin_blocks if cabin == cabin_block])
    return sum_of_fares / number_of_cabins

cabin_averages = [find_average_fare_for_cabin_block(cabin) for cabin in cabin_blocks]

plt.bar(cabin_blocks, cabin_averages)
plt.title('Average Fare Cost Per Cabin Block')
plt.xlabel('Cabin Block')
plt.ylabel('Fare Cost')


# In[ ]:


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(train_df['cabin_blocks']))
encoded_cabin_blocks = label_encoder.transform(list(train_df['cabin_blocks']))
print('Pearson Correlation Coefficient of Fare and Cabin Block: {}'.format(round(train_df['Fare'].corr(pd.Series(encoded_cabin_blocks)), 3)))


# The Pearson correlation coefficient does not suggest any strong correlation between fare and cabin block (moderate at best) so we cannot use fare to fill the `NaN` values in the cabin block column. Let's check if cabin block is correlated at all with survival.

# In[ ]:


print('Pearson Correlation Coefficient of Cabin Block and Survival: {}'.format(round(train_df['Survived'].corr(pd.Series(encoded_cabin_blocks)), 3)))


# There is a slight negative correlation between cabin block and survival, so we will include this feature for now. Note that survial and cabin are less correlated than fare and cabin. If it proves to be unuseful, then we will dampen it with regularization. 

# ### Task: Create family size feature
# 
# Family size is equal to `Parch` plus `SibSp`.

# In[ ]:


train_df['family_size'] = [row['Parch'] + row['SibSp'] for _, row in train_df.iterrows()]
train_df.head()


# In[ ]:


print('Pearson Correlation of Family Size and Survived: {}'.format(round(train_df['family_size'].corr(train_df['Survived']), 3)))


# In[ ]:


train_df['is_alone'] = [1 if size == 0 else 0 for size in train_df['family_size']]
train_df.head()


# In[ ]:


print('Pearson Correlation of Alone and Survived: {}'.format(round(train_df['Survived'].corr(train_df['is_alone']), 3)))


# `is_alone` is a slightly better feature than family size, but both have a relatively low correlation to `survived`.

# ### Task: Encode remaining features
# 
# We can drop `name`, `ticket`, and `cabin`. We need to encode `sex` and `embarked`.

# In[ ]:


train_df['cabin_blocks'] = encoded_cabin_blocks
label_encoder.fit(list(train_df['Sex']))
train_df['Sex'] = label_encoder.transform(list(train_df['Sex']))
label_encoder.fit(list(train_df['Embarked']))
train_df['Embarked'] = label_encoder.transform(list(train_df['Embarked']))
train_df.head()


# In[ ]:


sns.heatmap(train_df.corr(), 
            xticklabels=train_df.corr().columns.values,
            yticklabels=train_df.corr().columns.values)


# From the correlation matrix we note that `fare` and `sex` are highly correlated with `survived`.

# ### Prediction: Data setup
# We need to split the data into an additional training and test set to validate our results. **

# In[ ]:


train_df.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])


# ### Prediction: Random Forest
# We have a lot of categorical variables, so random forest is our best first choice for this data set.

# In[ ]:


y = train_df['Survived']
X = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'])
imput = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
imput = imput.fit(X)
X = imput.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[ ]:


clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

