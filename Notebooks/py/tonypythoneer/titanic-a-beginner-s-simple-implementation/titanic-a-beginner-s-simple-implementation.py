#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import basic packages

from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

import seaborn as sns

pd.set_option('display.max_rows', 10)


# In[2]:


# Files

train_file = '../input/train.csv'
test_file = '../input/test.csv'


# In[3]:


train_and_test_files = (train_file, test_file)
train_df, test_df = [pd.read_csv(f, index_col='PassengerId') for f in (train_file, test_file)]


# In[4]:


print('train_df:')
train_df.info()
print('---')
print('test_df:')
test_df.info()


# In[5]:


Correlation=train_df.corr()
print(pd.DataFrame(Correlation))

correlation_Y = pd.DataFrame(Correlation["Survived"])
correlation_Y.sort_values(by = "Survived", ascending = False)
print(correlation_Y)


# In[6]:


test_df.hist()


# # Check which columns are missing
# To enhance the completion of data. We should fill the missing data.

# In[7]:


unnecessary_columns = ['Ticket', 'Cabin']
train_df = train_df.drop(columns=unnecessary_columns)
test_df = test_df.drop(columns=unnecessary_columns)
merged_df = train_df.append(test_df)


# In[8]:


# Sould check what columns are including NaN
columns_series = merged_df.isnull().any()
columns_with_nan_series = columns_series[columns_series == True]
columns_with_nan = columns_with_nan_series.index.values.tolist()
columns_with_nan


# In[ ]:





# # Fill nan value of Fare column

# In[9]:


for c in columns_with_nan:
    subset_df = merged_df[merged_df[c].isnull()]
    row_count = len(subset_df.index)
    print('{} column has {} of row count of NaN data '.format(c, row_count))
    subset_df.head()


# In[10]:


p_class_types = merged_df['Pclass'].unique()
p_class_types.sort()
for p_class_type in p_class_types:
    fare_series = merged_df[merged_df['Pclass'] == p_class_type]['Fare']
    median = fare_series.median()
    print('Median fare of {} Pclass is: {}'.format(p_class_type, median))


# In[11]:


for p_class_type in p_class_types:
    fare_series = merged_df[merged_df['Pclass'] == p_class_type]['Fare']
    has_any_null = fare_series.isnull().any()
    if not has_any_null:
        continue

    s = fare_series.isnull()
    filled_series = s[s == True]
    merged_df.loc[filled_series.index, ['Fare']] = fare_series.median()


# In[12]:


embarked_count_series = merged_df['Embarked'].value_counts()


# # Fill nan value of Age column

# In[13]:


median_age = merged_df['Age'].median()
merged_df['Age'] = merged_df['Age'].fillna(median_age)


# # Fill nan value of Embarked column

# In[14]:


idxmax = merged_df['Embarked'].value_counts().idxmax()
print("Embarked 最常出現:", idxmax)
merged_df['Embarked'] = merged_df['Embarked'].fillna(idxmax)


# # Feature engineering about Name

# In[15]:


last_name_series = merged_df['Name'].str.split(", ", expand=True)[1]
last_name_series.head(5)


# In[16]:


title_series = last_name_series.str.split('.', expand=True)[0]


# In[17]:


merged_df['title'] = title_series
merged_df = merged_df.drop(columns=['Name'])
merged_df.head(5)


# In[18]:


pd.crosstab(merged_df['title'], merged_df['Sex']).T.style.background_gradient(cmap='summer_r')


# In[19]:


'''
Please refer here:
Mr. on wiki: https://en.wikipedia.org/wiki/Mr.
Miss on wiki: https://en.wikipedia.org/wiki/Miss
Ms on wiki: https://en.wikipedia.org/wiki/Ms.
'''
title_map = {
    'Capt': 'Mr',
    'Col': 'Mr',
    'Don': 'Mr',
    'Dona': 'Mrs',
    'Dr': 'Mr',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Major': 'Mr',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Ms': 'Miss',
    'Rev': 'Mr',
    'Sir': 'Mr',
    'the Countess': 'Mrs',
    'Master': 'Mr'}
merged_df['title'] = merged_df['title'].replace(title_map)


# In[20]:


dummy = pd.get_dummies(merged_df['title'])
merged_df = pd.concat([merged_df, dummy], axis=1)
merged_df = merged_df.drop(columns=['title'])


# In[21]:


merged_df.head()


# # Convert Sex, Embarked and Pclass to 0/1 variables
# 
# We should use get_dummies to handle this<br>
# Convert categorical variable into dummy/indicator variables

# In[22]:


for column in ['Sex', 'Embarked']:
    dummy = pd.get_dummies(merged_df[column])
    merged_df = pd.concat([merged_df, dummy.astype(bool)], axis=1)
    merged_df = merged_df.drop([column], axis=1)
    
dummy = pd.get_dummies(merged_df['Pclass'], prefix='pclass')
merged_df = pd.concat([merged_df, dummy.astype(bool)], axis=1)
merged_df = merged_df.drop(['Pclass'], axis=1)


# In[23]:


merged_df.head()


# # Merge Parch and SibSp into family

# In[24]:


merged_df['family'] = merged_df['Parch'] + merged_df['SibSp']
merged_df = merged_df.drop(columns=['Parch', 'SibSp'])
merged_df['family'] = merged_df['family'].astype(int)
merged_df.head()


# In[25]:


new_train_df = merged_df[pd.notnull(merged_df['Survived'])]


# In[26]:


plt.figure(figsize=(14, 14))
sns.heatmap(merged_df.astype(float).corr(), cmap = 'BrBG',
            linewidths=0.1, square=True, linecolor='white',
            annot=True)


# # Final

# In[27]:


new_train_df = merged_df[merged_df['Survived'].notnull()]
new_test_df = merged_df[merged_df['Survived'].isnull()]


# In[28]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


# In[29]:


clf = DecisionTreeClassifier(max_depth=3)
scores = cross_val_score(clf,
                        new_train_df.drop(['Survived'], axis=1), 
                        new_train_df['Survived'],
                        cv=10)
scores


# In[30]:


scores.mean()


# In[31]:


clf.fit(new_train_df.drop(['Survived'], axis=1), new_train_df['Survived'])


# In[32]:


from sklearn.tree import export_graphviz
import graphviz
g = export_graphviz(clf,out_file=None,
                    feature_names=new_train_df.drop(['Survived'], axis=1).columns,
                    class_names=["No", "Yes"],
                    filled=True, 
                    rounded=True,
                    special_characters=True)


# In[33]:


graphviz.Source(g)


# In[34]:


pprint(dict(zip(new_train_df.drop(['Survived'], axis=1).columns.tolist(), clf.feature_importances_)))


# In[35]:


from sklearn.ensemble import RandomForestClassifier


# In[36]:


from sklearn.model_selection import cross_val_score


# In[37]:


x_train = new_train_df.drop(['Survived'], axis=1)
x_test = new_train_df['Survived']


# In[54]:


scores = []
for n_estimators in range(10, 110, 5):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(x_train, x_test)
    score = clf.score(x_train, x_test)
    scores.append(score)


# In[55]:


plt.plot(range(10, 110, 5), scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()


# In[40]:


pprint(dict(zip(range(10, 110, 5), scores)))


# In[41]:


scores = []
for n_estimators in range(25, 36):
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf = clf.fit(x_train, x_test)
    score = clf.score(x_train, x_test)
    scores.append(score)


# In[42]:


plt.plot(range(25, 36), scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.show()


# In[43]:


pprint(dict(zip(range(25, 36), scores)))


# In[44]:


scores = []
for max_depth in range(3, 21):
    clf = RandomForestClassifier(n_estimators=26, max_depth=max_depth)
    clf = clf.fit(x_train, x_test)
    score = clf.score(x_train, x_test)
    scores.append(score)


# In[45]:


plt.plot(range(3, 21), scores)
plt.xlabel('max_depth')
plt.ylabel('score')
plt.show()


# In[46]:


pprint(dict(zip(range(3, 21), scores)))


# In[47]:


clf = RandomForestClassifier(n_estimators=26, max_depth=15)
clf = clf.fit(x_train, x_test)


# In[48]:


predict_result = clf.predict(new_test_df.drop(['Survived'], axis=1))


# In[49]:


new_test_df['Survived'] = predict_result


# In[50]:


survived = new_test_df.loc[:,['Survived']]
survived['Survived'] = survived['Survived'].astype(int) 
#survived = new_test_df['Survived']


# In[51]:


survived.to_csv('submission.csv')


# In[ ]:




