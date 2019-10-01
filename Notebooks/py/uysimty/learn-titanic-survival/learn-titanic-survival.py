#!/usr/bin/env python
# coding: utf-8

# # Titanic Model
# This is my first Kernel. I use **RandomForestClassifier** for as the model.
# I am happy to get comment to improve this Kernel.

# # Libraries

# In[ ]:


import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# # Import Data

# In[ ]:


raw_data = pd.read_csv("../input/train.csv")
raw_test = pd.read_csv('../input/test.csv')


# # Data Exploration

# **Data Columns**

# In[ ]:


print(raw_data.columns)


# **Head Of Data**

# In[ ]:


print(raw_data.head())


# **Data Types**

# In[ ]:


raw_data.info()


# **First Row**

# In[ ]:


print(raw_data.iloc[10])


# **Select Valid Data**

# In[ ]:


need_columns = ['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare']
data = raw_data[need_columns]

# Convert Male and Female To number (0, 1)
gender_encoder = LabelEncoder()
data['Sex'] = gender_encoder.fit_transform(data['Sex'])

# Get ride of NAN
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
data = imputer.fit_transform(data)

x = data
y = raw_data.Survived


# **Check Null Data**

# In[ ]:


print(raw_data.isnull().sum())
print("-"*10)
print(raw_data.isnull().sum()/raw_data.shape[0])


# We can see there are a lot of people that don't have Cabin record. So we won't use this colum

# **Corralation Of Data**

# In[ ]:


cor_matrix = raw_data.drop(columns=['PassengerId']).corr().round(2)
# Plotting heatmap 
fig = plt.figure(figsize=(12,12));
sns.heatmap(cor_matrix, annot=True, cmap='autumn');


# **Survival Count**

# In[ ]:


sns.countplot(x='Survived', data=raw_data)


# **Survival By Sex**

# In[ ]:


sns.barplot(x='Sex', y='Survived', data=raw_data)


# In[ ]:


sns.countplot(x="Sex", hue="Survived", data=raw_data)


# We can see female have more chance to survive than man. As we know they evaluate female first.

# **Survival By Class**

# In[ ]:


sns.barplot(x='Pclass', y='Survived', data=raw_data)


# In[ ]:


sns.countplot(x="Pclass", hue="Survived", data=raw_data)


# The people who say at upper class have more chance to survived

# **Plot Ages**

# In[ ]:


plt.figure(figsize=(18, 30))
sns.countplot(y='Age', data=raw_data)


# **Survival by Age**

# In[ ]:


raw_data['Age'] = raw_data['Age'].dropna().astype(int)
sns.FacetGrid(raw_data, hue='Survived', aspect=4).map(sns.kdeplot, 'Age', shade= True).set(xlim=(0 , raw_data['Age'].max())).add_legend()


# **Age by group every 10 year old**

# In[ ]:


raw_data['AgeGroup'] = pd.cut(raw_data.Age, bins=16)
plt.figure(figsize=(18, 5))
sns.barplot(x='AgeGroup', y='Survived', data=raw_data)


# **Train Model**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)


# Graph of survive and unsurvive is almost similar. So age is not the reason people survive. But even so we can see that children under 5 year old have chance to survived

# In[ ]:


raw_data['IsChildren'] = np.where(raw_data['Age']<=5, 1, 0)
sns.countplot(x='Survived', data=raw_data[raw_data.IsChildren==1])


# **Survival by number of sibling or spouse**

# In[ ]:


plt.figure(figsize=(18, 8))
sns.barplot(x='SibSp', y='Survived', data=raw_data)


# **Survival by number of parent or children**

# In[ ]:


plt.figure(figsize=(18, 8))
sns.barplot(x='Parch', y='Survived', data=raw_data)


# **Family Size**

# In[ ]:


plt.figure(figsize=(18, 8))
raw_data['FamilySize'] = raw_data.apply (lambda row: row['SibSp']+row['Parch'], axis=1)
sns.barplot(x='FamilySize', y='Survived', data=raw_data)


# From this we can see who travel with 1, 2 or 3 people have more chance to survive

# **Survival by fare**

# In[ ]:


plt.subplots(1,1,figsize=(18, 8))
sns.distplot(raw_data['Fare'].dropna())


# We need to group fare by each 50 dollar

# In[ ]:


raw_data['FareRange'] = pd.cut(raw_data.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, include_lowest=True)
raw_data['FareGroup'] = pd.cut(raw_data.Fare, bins=np.arange(start=0, stop=600, step=50), precision=0, include_lowest=True, labels=False)
plt.figure(figsize=(18, 8))
sns.barplot(x='FareRange', y='Survived', data=raw_data)


# In[ ]:


plt.figure(figsize=(18, 8))
sns.countplot('FareRange', hue='Survived', data=raw_data)


# We can see that people likely to survive if Fare is bigger

# In[ ]:


raw_data['LowFare'] = np.where(raw_data['Fare']<=50, 1, 0)
sns.countplot('LowFare', hue='Survived', data=raw_data)


# **Survival by Embarked**

# In[ ]:


sns.barplot(x='Embarked', y='Survived', data=raw_data)


# People Cherbourg from get a lot of chance to survive

# **Embarked And Sex**

# In[ ]:


plt.figure(figsize=(18, 8))
sns.FacetGrid(raw_data,size=5, col="Sex", row="Embarked", hue = "Survived").map(plt.hist, "Age", edgecolor = 'white').add_legend();


# * We can see most of the passengers boarding at Southampton. And all of them mostly male who less likely to survived
# * Most of passengers from Cherbourg is female that why they have a lot of percentage to survived
# * Few female boarding at Queenstown and most of them survived
# 
# So we can Embarked is not related to the survived.

# **Title**

# In[ ]:


raw_data['Title'] = raw_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
plt.figure(figsize=(18, 8))
print(raw_data['Title'].unique())

raw_data['Title'] = raw_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
raw_data['Title'] = raw_data['Title'].replace('Mlle', 'Miss')
raw_data['Title'] = raw_data['Title'].replace('Ms', 'Miss')
raw_data['Title'] = raw_data['Title'].replace('Mme', 'Mrs')

sns.countplot(x="Title", hue="Survived", data=raw_data)


# # Data Preparation

# In[ ]:


from sklearn.model_selection import train_test_split
need_columns = ['Pclass', 'Sex', 'IsChildren', 'FamilySize', 'LowFare', 'Title']
data = raw_data[need_columns]

# Convert Male and Female To number (0, 1)
from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
data['Sex'] = gender_encoder.fit_transform(data['Sex'])

title_encoder = LabelEncoder()
data['Title'] = title_encoder.fit_transform(data['Title'])
x = data
y = raw_data.Survived

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


# # Find Best Params

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
max_leaf_nodes = [2, 5, 8, 10, None]
criterion=['gini', 'entropy']

random_grid = {
    'n_estimators': n_estimators,
    'criterion': criterion,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap,
  'max_leaf_nodes': max_leaf_nodes
}

estimator = RandomForestClassifier()
rf_random = RandomizedSearchCV(
    estimator=estimator, 
    param_distributions=random_grid, 
    n_iter=100, 
    cv=3, 
    random_state=42, 
    n_jobs=-1
)
rf_random.fit(x, y)

best_params = rf_random.best_params_
print(best_params)


# # Testing

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

test_model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    criterion=best_params['criterion'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    bootstrap=best_params['bootstrap']
)

test_model.fit(x_train, y_train)
predicted_y = test_model.predict(x_test)
print("Error: {}".format(mean_absolute_error(y_test, predicted_y)))
print("Accuracy: {}".format(accuracy_score(y_test, predicted_y)))


# # Training

# In[ ]:


model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    criterion=best_params['criterion'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    bootstrap=best_params['bootstrap']
)
model.fit(x, y)


# # Submit Result

# In[ ]:


raw_test['IsChildren'] = np.where(raw_test['Age']<=5, 1, 0)
raw_test['FamilySize'] = raw_test.apply (lambda row: row['SibSp']+row['Parch'], axis=1)
raw_test['LowFare'] = np.where(raw_test['Fare']<=50, 1, 0)

raw_test['Title'] = raw_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
raw_test['Title'] = raw_test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
raw_test['Title'] = raw_test['Title'].replace('Mlle', 'Miss')
raw_test['Title'] = raw_test['Title'].replace('Ms', 'Miss')
raw_test['Title'] = raw_test['Title'].replace('Mme', 'Mrs')

data_test = raw_test[need_columns]
data_test['Sex'] = gender_encoder.transform(data_test['Sex'])
data_test['Title'] = title_encoder.transform(data_test['Title'])

ids = raw_test['PassengerId']
predictions = model.predict(data_test)

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
print(output.head())
output.to_csv('submission.csv', index = False)

