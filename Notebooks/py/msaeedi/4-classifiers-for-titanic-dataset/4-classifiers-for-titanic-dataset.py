#!/usr/bin/env python
# coding: utf-8

# The goal of this notebook is to try a 4 simple classifiers (svm, mlp, decision tree and random forest) for the [Kaggle's Titanic: Machine Learning from Disaster dataset](https://www.kaggle.com/c/titanic). I use Python for this project. First, let's start by reading the dataset to see what we have:

# In[ ]:


import pandas as pd
directory = '../input/'
titanic_train = pd.read_csv(directory + 'train.csv')
titanic_test = pd.read_csv(directory + 'test.csv')


# In[ ]:


titanic_train.info()


# **Missing Data**
# 
# There are some missing data in Age, Cabin, and Fare as you can see below. 

# In[ ]:


titanic_train.isnull().sum()


# In[ ]:


titanic_test.isnull().sum()


# Let's combine the training and test data and fill the missing data. For now, I'll use simple methods. We can improve them later. 

# In[ ]:


titanic_train_test = [titanic_train, titanic_test]


# In[ ]:


for dataset in titanic_train_test:
    dataset['Age'].fillna(dataset.Age.median(), inplace=True)
    dataset['Cabin'].fillna('U', inplace=True)
    dataset['Embarked'].fillna('S', inplace=True)
    dataset['Fare'].fillna(dataset.Fare.mean(), inplace=True)


# In[ ]:


titanic_train.info()
titanic_test.info()


# **Adding new features**
# 
# I'm going to define a few more columns for now. I define FamilySize (based on SibSp:# of siblings / spouses aboard the Titanic and Parch (of parents / children aboard the Titanic). 

# In[ ]:


for dataset in titanic_train_test:
    dataset['FamilySize'] = dataset.SibSp + dataset.Parch


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
titanic_train.groupby('Age').count().PassengerId.plot()


# It's better to group the ages so we have a better understanding of the groups. 

# In[ ]:


for dataset in titanic_train_test:
    dataset['AgeRange'], AgeBins = pd.cut(dataset['Age'], 10, retbins=True)


# In[ ]:


titanic_train.groupby('AgeRange').count().PassengerId.plot.barh()


# In[ ]:


titanic_train.Name.head(10)


# We can extract titles and then use it instead of Sex. Actually, there is a clear correlation between Sex and Title (Male cannot use Mrs for title). So, it doesn't make sense to keep both. As the same time, I think Title shows more information than a binary variable Sex.

# In[ ]:


for dataset in titanic_train_test:
    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())


# In[ ]:


titanic_train.Title.head(10)


# In[ ]:


titanic_train.groupby('Title').count().PassengerId


# In[ ]:


titanic_test.groupby('Title').count().PassengerId


# It make sense to just keep a few titles and get rid of the rare cases. 

# In[ ]:


for dataset in titanic_train_test:
   dataset['Title'] = dataset['Title'].map(
    lambda x: 'unusual' if x in ["Capt", "Col", "Major", "Jonkheer", "Don", "Sir", "Dr", "Rev", "the Countess", "Dona",
                              "Lady"] else ('Mrs' if x in ["Mme", "Ms"] else ('Miss' if x in ["Mlle"] else x)))


# In[ ]:


plt.subplot(1,2,1)
titanic_train.groupby('Title').count().PassengerId.plot.bar()
plt.subplot(1,2,2)
titanic_test.groupby('Title').count().PassengerId.plot.bar()


# Let's see what we can get from last name. Basically, if we can use the same last name to distinguish people from the same family, it can be useful. The problem is that some last names are common and we should not use them (see max value is 9 in the training data).

# In[ ]:


for dataset in titanic_train_test:
    dataset['LastName'] = dataset['Name'].map(lambda name: name.split(',')[0].strip())


# In[ ]:


titanic_train.groupby('LastName').count().PassengerId.describe()


# In[ ]:


titanic_test.groupby('LastName').count().PassengerId.describe()


# Let's investigate what we have from tickets, cabins, and fares. Remeber, 'U' in Cabin means unknown. The 687 value in the training data is for that one. For ticket, things are better. The max number of items with the same ticket is 7. Maybe we can use it to find groups. Obviously, same fare does not mean anything is most cases. But, it we have very few persons with the same fare, maybe they got their tickets together.

# In[ ]:


titanic_train.groupby('Cabin').count().PassengerId.describe()


# In[ ]:


titanic_train.groupby('Ticket').count().PassengerId.describe()


# In[ ]:


titanic_train.groupby('Fare').count().PassengerId.describe()


# I think we can find the frequencies of things now to see what are the distributions. 

# In[ ]:


for dataset in titanic_train_test:
    for col in ['Ticket', 'Cabin', 'Fare', 'LastName']:
        freq_col = f'Freq{col}'

        freq = dataset[col].value_counts().to_frame()
        freq.columns = [freq_col]

        dataset[freq_col] = dataset.merge(freq, how='left', left_on=col, right_index=True)[freq_col]


# In[ ]:


titanic_train.info()


# In[ ]:


titanic_train.groupby('FreqTicket').count().PassengerId.plot.bar()


# In[ ]:


titanic_train.groupby('FreqCabin').count().PassengerId.plot.bar()


# In[ ]:


titanic_train.groupby('FreqLastName').count().PassengerId.plot.bar()


# In[ ]:


titanic_train.groupby('FreqFare').count().PassengerId.plot.barh()


# Now, we can group things together. I'm going to use FamilySize first, then FreqTicket, then FreqCabin and then FreqLastName. If there is nothing, then the passanger was alone.

# In[ ]:


def groupify(x):
    max_group = 5
    if x['FamilySize'] > 0:
        return x['FamilySize']
    elif x['FreqTicket'] > 1:
        return x['FreqTicket']
    elif x['FreqCabin'] > 1 and x['Cabin'] != 'U':
        return x['FreqCabin']
    elif 1 < x['FreqLastName'] < max_group:
        return x['FreqLastName']
    elif 1 < x['FreqFare'] < max_group:
        return x['FreqFare']
    else:
        return 0


# In[ ]:


for dataset in titanic_train_test:
    dataset['GroupSize'] = dataset.apply(groupify, axis=1)


# In[ ]:


titanic_train.groupby('GroupSize').count().PassengerId.plot.bar()


# **Final set of features**
# 
# Let's see what we got finally:

# In[ ]:


print(titanic_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print()
print(titanic_train[['GroupSize', 'Survived']].groupby(['GroupSize'], as_index=False).mean())
print()
print(titanic_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
print()
print(titanic_train[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean())
print()
print(titanic_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
print()


# Let's remove the ununsed columns and clean the final dataset.

# In[ ]:


y = titanic_train['Survived']
titanic_train.drop(['Survived'], axis=1, inplace=True)


# In[ ]:


for dataset in titanic_train_test:
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "unusual": 5}).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    for AgeGroup in range(0, len(AgeBins)):
        if AgeGroup == len(AgeBins) - 1:
            dataset.loc[dataset['Age'] > AgeBins[AgeGroup], 'Age'] = AgeGroup
        else:
            dataset.loc[
                (dataset['Age'] > AgeBins[AgeGroup]) & (dataset['Age'] <= AgeBins[AgeGroup + 1]), 'Age'] = AgeGroup

    dataset["Pclass"] = dataset["Pclass"].astype('int')

    # Sex & Title have correclation. We keep Title.
    for col in dataset.columns:
        if col not in ['Pclass', 'Age', 'Embarked', 'Title', 'GroupSize']:
            dataset.drop([col], inplace=True, axis=1)
    for col in dataset.columns:
        dataset[col] = dataset[col].astype("category")


# In[ ]:


titanic_train.columns


# **Converting to 0/1 Features**
# 
# As you can see, I almost remove all columns and kept the very few. I beleive that other columns are very related to the above ones and as such there will be correlation between them.
# 
# I'm going to make binary variables from these columns.

# In[ ]:


titanic_train = pd.get_dummies(titanic_train, columns=None)
titanic_test = pd.get_dummies(titanic_test, columns=None)


# In[ ]:


titanic_train.info()


# To make sure we are using the same feature sets for both train and test, I need to clean the dataset a little more.

# In[ ]:


missing_cols = set(titanic_train.columns) - set(titanic_test.columns)
for c in missing_cols:
    titanic_test[c] = 0
missing_cols = set(titanic_test.columns) - set(titanic_train.columns)
for c in missing_cols:
    titanic_test[c] = 0


# In[ ]:


X_train, y_train = titanic_train, y
X_test = titanic_test


# **Cross Validation with different classifiers**
# 
# I'm going to try a few classifiers with different paramters (cross validation)

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

# Set the parameters by cross-validation
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy as np

# run svm
param_grid = {"gamma": np.logspace(-3, 3, 7),
              "C": np.logspace(-3, 3, 7)              }
svm_model = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
svm_model.fit(X_train, y_train)


# In[ ]:


print("[SVM] The best parameters are %s with a score of %0.2f"
      % (svm_model.best_params_, svm_model.best_score_))


# Now, let's try Multi-layer Perceptron and Decision Trees.

# In[ ]:


from sklearn.neural_network import MLPClassifier

# MLP
param_grid = {"hidden_layer_sizes": [(50,), (50, 50)],
              "alpha": np.logspace(-3, 3, 7)
              }
mlp = GridSearchCV(MLPClassifier('lbfgs', max_iter=600), param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
mlp.fit(X_train, y_train)


# In[ ]:


print("[MLP] The best parameters are %s with a score of %0.2f"
      % (mlp.best_params_, mlp.best_score_))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# Tree
param_grid = {"max_depth": np.linspace(10, 15, 6).astype(int),
              "min_samples_split": np.linspace(2, 5, 4).astype(int)
              }
clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=cv)
clf.fit(X_train, y_train)


# In[ ]:


print("[TREE] The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))


# In[ ]:


importances = clf.best_estimator_.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=[16, 8])
plt.title('Feature Importances for DecisionTreeClassifier')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.xlabel('Relative Importance')


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Random Forest
param_grid = {"n_estimators": [250, 300],
              "criterion": ["gini", "entropy"],
              "max_depth": [10, 15, 20],
              "min_samples_split": [2, 3, 4]}
forest = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, verbose=1)
forest.fit(X_train, y_train)


# In[ ]:


print("[FOREST] The best parameters are %s with a score of %0.2f"
      % (forest.best_params_, forest.best_score_))


# In[ ]:


importances = forest.best_estimator_.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=[16, 8])
plt.title('Feature Importances for RandomForestClassifier')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), X_train.columns[indices])
plt.xlabel('Relative Importance')


# So, it seems the random forest is the best one.  Now, we can stack the models discussed above to potentially improve the prediction. I use the simple paramters based on the experiments we did above. We can run GridSearchCV in combination with the stacked model as well. 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingCVClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split


clf1 = svm.SVC(C=1, gamma=0.1)
clf2 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=600, alpha=1)
clf3 = DecisionTreeClassifier(max_depth=10, min_samples_split=4)
clf4 = RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_split=4, criterion='gini')
lr = LogisticRegression()

sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4],
                            meta_classifier=lr)

X_train2, X_cv, y_train2, y_cv = train_test_split(X_train, y_train, test_size=0.33, random_state=0)

sclf.fit(X_train2.values, y_train2.values)
print("[Stacking] score on training data is %0.2f", sclf.score(X_train2.values, y_train2.values))
print("[Stacking] score on the crossvalidation data is %0.2f", sclf.score(X_cv.values, y_cv.values))

