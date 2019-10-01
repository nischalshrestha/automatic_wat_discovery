#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')


# In[ ]:


train_dataset.head()


# In[ ]:


train_dataset.info()


# From the info function we can see following
# - There are 891 entries in total
# - Age, Cabin, Embarked have nan values

# In[ ]:


test_dataset.head()


# In[ ]:


test_dataset.info()


# From the info function we can see following
# - There are 418 entries in total
# - Age, Fare, Cabin have nan values

# In[ ]:


train_dataset[train_dataset['Cabin'].isnull()]


# In[ ]:


test_dataset[test_dataset['Cabin'].isnull()]


#     We will drop PassengerId column as that is just a counter and does not impact survival rate.
#     We will drop Ticket as that column should not have any correlation with survival
#     We will also drop Cabin column as many of the values are nulls
#     We will separate other parameter and label (Survived in this case)

# In[ ]:


X_train = train_dataset.drop(columns=['PassengerId', 'Survived', 'Cabin', 'Ticket'])
y_train = train_dataset['Survived']
X_test = test_dataset.drop(columns=['PassengerId', 'Cabin', 'Ticket'])
X_all = pd.concat([X_train,X_test], axis=0)


# In[ ]:


X_all.info()


# We have merged parameter data from train and test so that we can identify which all columns have NaNs and then take care of those NaN values.

# In[ ]:


X_all[X_all['Embarked'].isnull()]


# Both null values are for Pclass = 1 and Fare 80.0

# In[ ]:


import seaborn as sns
plt.figure(figsize=(10, 5))
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=X_all);


# We could see that for Embarcked = 'C', mean fare for class 1 is somewhere around 80.0
# So we will fill 'C' for these records in the Embarked column.

# In[ ]:


X_all['Embarked'] = X_all['Embarked'].fillna('C')


# In[ ]:


X_all[X_all['Fare'].isnull()]


# Get the median fare value for Pclass = '3' and Embarked = 'S'

# In[ ]:


X_all_3_S_median = X_all[(X_all['Embarked'] == 'S') & (X_all['Pclass'] == 3)]['Fare'].median()
print("Median value of class 3 and embarked s ==> " + str(X_all_3_S_median))


# In[ ]:


X_all['Fare'] = X_all['Fare'].fillna(X_all_3_S_median)


# In[ ]:


X_all.info()


# Now olny column with NaNs is Age. 
# To fill in the Age column we will use RandomForest regressor.
# But first we need to encode all the categorical columns.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_all.iloc[:,2] = le.fit_transform(X_all.iloc[:,2])
# Female = 0, Male = 1
le = LabelEncoder()
X_all.iloc[:,7] = le.fit_transform(X_all.iloc[:,7])
# C = 0, Q = 1, S = 2


# In[ ]:


X_all.head()


# In[ ]:


fig = plt.figure(figsize=(18,6))
train_dataset.Survived.value_counts().plot(kind='bar')


# In[ ]:


train_dataset.Survived.value_counts(normalize=True).plot(kind='bar')


# In[ ]:


X_all.hist(bins=10,figsize=(9,7))


# In[ ]:


import seaborn as sns
g = sns.FacetGrid(train_dataset, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",  color="purple");


# We can see from above that 'sex' plays important role in determining survival.

# In[ ]:


g = sns.FacetGrid(train_dataset, hue="Survived", col="Pclass", row="Sex", margin_titles=True,
                  palette={1:"green", 0:"red"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();


# We can also see that 'Fare' plays critical role. 

# In[ ]:


corr = train_dataset.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# In[ ]:


train_dataset.corr()['Survived']


# We can see that there is high correlation (positive or negative) between survived and 
# - Pclass
# - Fare
# - Parch
# - Age
# (in this order)

# Age of a person might be dependent on Salutation of a person 
# 
# we will need to construct the Salutaion of the person.
# Name is in the format ==> Lastname, Salutation. Firstname ...

# In[ ]:


X_all['Salutation'] = X_all.apply(lambda row: row['Name'].split()[1], axis=1)


# In[ ]:


X_all.iloc[:,8] = le.fit_transform(X_all.iloc[:,8])


# In[ ]:


X_all.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
#predicting missing values in age using Random Forest
def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Salutation']]
    # Split sets into train and test
    train  = age_df.loc[ (age_df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (age_df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])
    
    # Assign those predictions to the full data set
    df.loc[ (age_df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df


# In[ ]:


X_all=fill_missing_age(X_all)


# In[ ]:


X_all.info()


# In[ ]:


X_train = X_all.iloc[:891, [0,2,3,4,5,6,7,8]]
X_test = X_all.iloc[891:,[0,2,3,4,5,6,7,8]]
X_train.info()
X_test.info()


# In[ ]:


from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0, n_jobs = -1)
lr_classifier.fit(X_train, y_train)
lr_y_pred = lr_classifier.predict(X_test)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_jobs = -1)
knn_classifier.fit(X_train, y_train)
knn_y_pred = knn_classifier.predict(X_test)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 2000, criterion='entropy', 
                                       n_jobs=-1, random_state = 100)
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(X_train, y_train)
dt_y_pred = dt_classifier.predict(X_test)


# In[ ]:


from sklearn.svm import SVC
sc_classifier = SVC(cache_size = 3000)
sc_classifier.fit(X_train, y_train)
svc_y_pred = sc_classifier.predict(X_test)


# In[ ]:


from statistics import mode
final_pred = []
for i in range(418):
    final_pred.append(mode([lr_y_pred[i],
                           knn_y_pred[i],
                           rf_y_pred[i],
                           dt_y_pred[i],
                           svc_y_pred[i]]))


# In[ ]:


final_pred

