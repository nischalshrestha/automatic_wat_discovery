#!/usr/bin/env python
# coding: utf-8

# In[431]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

cmap = sns.diverging_palette(250, 10, as_cmap=True)


# In[432]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
datasets = [train, test]


# In[433]:


train.head()


# In[434]:


test.head()


# In[435]:


train.info()
print('_'*40)
test.info()


# In[436]:


train.isnull().sum()


# In[437]:


test.isnull().sum()


# In[438]:


train.dtypes


# In[439]:


train.describe()


# In[440]:


test.describe()


# In[441]:


plt.subplots(figsize=(12,9))
sns.heatmap(train.drop(["PassengerId"], axis = 1).corr(), annot = True, cmap = cmap)


# In[442]:


for dataset in datasets:
    dataset.drop(["Cabin"], axis = 1, inplace = True)


# In[443]:


train["Embarked"].value_counts()


# In[444]:


for dataset in datasets:
    dataset["Embarked"].fillna("S", inplace = True)


# In[445]:


train.info()
print('_'*40)
test.info()


# In[446]:


train[["Sex", "Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived", ascending = False)


# In[447]:


train[["Pclass", "Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Pclass")


# In[448]:


train[["Embarked", "Survived"]].groupby(["Embarked"], as_index = False).mean().sort_values(by = "Embarked")


# In[449]:


pd.crosstab([train["Embarked"], train["Pclass"]], [train["Sex"], train["Survived"]], margins = True).style.background_gradient(cmap = cmap)


# In[450]:


for dataset in datasets:
    dataset["Embarked"] = dataset["Embarked"].map({"C": 0, "Q": 1, "S": 2})
train.head()


# In[451]:


for dataset in datasets:
    dataset["FamilySize"] = dataset["SibSp"]+dataset["Parch"]+1
train.head()


# In[452]:


pd.crosstab(train["FamilySize"], train["Survived"], margins = True).style.background_gradient(cmap = cmap)


# In[453]:


train.head()


# In[454]:


for dataset in datasets:
    dataset.drop(["Ticket"], axis = 1, inplace = True)
train.head()


# In[455]:


for dataset in datasets:
    dataset["Sex"] = dataset["Sex"].map({"male":0, "female":1})
train.head()


# In[456]:


for dataset in datasets:
    dataset["Title"] = dataset["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train.head()


# In[457]:


pd.crosstab(train["Title"], train["Sex"], margins = True).sort_values(by = "All", ascending = False)


# In[458]:


for dataset in datasets:
    dataset["Title"] = dataset["Title"].replace(["Dr", "Rev", "Major", "Col", "Mlle", "Don", "Jonkheer", "Lady", "Mme", "Countess", "Ms", "Sir", "Capt"], "Other")
    dataset["Title"] = dataset["Title"].map({"Mr":0, "Miss":1, "Mrs":2 , "Master": 3, "Other" :4})


# In[459]:


for dataset in datasets:
    dataset["Title"].fillna(0, inplace = True)


# In[460]:


for dataset in datasets:
    dataset.drop(["Name"], axis = 1, inplace = True)


# In[461]:


data = train.append(test)
titles = [0, 1, 2, 3, 4]
for title in titles:
    age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]
    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute
    
# Substituting Age values in TRAIN_DF and TEST_DF:
train['Age'] = data['Age'][:891]
test['Age'] = data['Age'][891:]

# Dropping Title feature
for dataset in datasets:
    dataset.drop(["Title"], axis = 1, inplace = True)


# In[462]:


data['AgeBin'] = pd.qcut(data['Age'], 4)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])

train['AgeBin_Code'] = data['AgeBin_Code'][:891]
test['AgeBin_Code'] = data['AgeBin_Code'][891:]

train.drop(['Age'], 1, inplace=True)
test.drop(['Age'], 1, inplace=True)


# In[463]:


data["Fare"].fillna(data["Fare"].median(), inplace = True)

# Making Bins
data['FareBin'] = pd.qcut(data['Fare'], 5)

label = LabelEncoder()
data['FareBin_Code'] = label.fit_transform(data['FareBin'])

train['FareBin_Code'] = data['FareBin_Code'][:891]
test['FareBin_Code'] = data['FareBin_Code'][891:]

train.drop(['Fare'], 1, inplace=True)
test.drop(['Fare'], 1, inplace=True)


# In[464]:


train.head()


# In[465]:


for dataset in datasets:
    dataset.drop(["Embarked"], axis = 1, inplace = True)
    dataset.drop(["Parch"], axis = 1, inplace = True)
    dataset.drop(["SibSp"], axis = 1, inplace = True)
train.head()


# In[466]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(["Survived"], axis = 1), train["Survived"])


# In[467]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100,
                                criterion='gini',
                                max_depth=5,
                                min_samples_split=10,
                                min_samples_leaf=5,
                                random_state=0)
classifier.fit(X_train, y_train)

print("Random Forest score: {0:.2}".format(classifier.score(X_test, y_test)))


# In[468]:


prediction = classifier.predict(test)

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})
submission.to_csv("submission.csv", index=False)


# In[ ]:




