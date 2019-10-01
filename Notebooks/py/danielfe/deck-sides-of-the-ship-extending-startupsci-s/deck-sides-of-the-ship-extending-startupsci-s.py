#!/usr/bin/env python
# coding: utf-8

# # This Titanic notebook will be the first time I solve kaggle by myselfe (kind of).
# ### This will be an attempt to improve https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook / do it in my own words.
# 
# ## Abstract:
# I will  solve the titanic challange from the Kaggle website.<br>
# This solution is based on a tutorial https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook, however it has two main differrences which eventually gave a higher score:<br>
# > - "Deck" feature: This is the main cause of improvement
# > - "CabinBand" feature: This might be a minor cause in improvement
# 
# ### The "Deck" feature
# In the tutorial, the author decided to ignore the "Cabin" feature, when I saw the cabin feature I noticed a pattern which might indicate the location of the rooms on the ship.<br>
# In order to investigate this I found a website with the titanics deckplans: https://www.encyclopedia-titanica.org/titanic-deckplans/ <br>
# In the deckplans you can see that there are more decks than classes, which might mean that the only reason that classes is a major player here is because it is located in higher decks. This further means that there are classes which the survival rate within the class changes according to the deck.<br>
# Luckily, the cabin number contains the deck as a prefixed letter to the cabin number, therefore I created a new feature out of this letter called "Deck". We will later see in the decision tree visualization that this feature played a major role in the classification.
# 
# ### The "CabinBand" Feature
# If you look at the deck plans, another feature you can derive from the cabin number is the approximate location on the plane of the deck.<br>
# unfortunately, when checking if there was any correlation between the different ends of the ship to the survivavl rate, I found no good correlation.
# However, when checking correlation between the different sides of the ship, I found a better correlation.<br>
# I decided to use this feature. It played a smaller role than the deck as you can imagine.

# ## Load the data

# In[ ]:


import pandas as pd

test_df = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')
combine_df = [train_df, test_df]
train_df.tail()


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# we will not need PassengereId

# In[ ]:


train_df.drop(["PassengerId"], axis=1, inplace=True)


# ## Handle Sex:

# In[ ]:


for dataset in combine_df:
    dataset.Sex.fillna(-1, inplace=True)
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train_df.head(10) 


# ## handle the Cabin feature

# ### convert Cabin into Cabin Number -> Cabin  Band

# In[ ]:


for dataset in  combine_df:
    dataset['CabinNumber'] = dataset.Cabin.str.extract('[A-G](\d+)', expand=False)
    dataset.CabinNumber.fillna(0, inplace=True)
    dataset['CabinNumber'] = dataset['CabinNumber'].astype(int)
combine_df = [train_df, test_df]
train_df.CabinNumber.unique()


# In[ ]:


import seaborn as sns
for dataset in combine_df:
    dataset['CabinBand'] = 0
    dataset.loc[dataset['CabinNumber'] == 0, 'CabinBand'] = 1
    dataset.loc[dataset['CabinNumber'] % 2 == 1, 'CabinBand'] = 2
    #dataset['CabinBand'] = pd.cut(dataset['CabinNumber'], 3, labels=['front', 'middle', 'back'])
    dataset.drop(['CabinNumber'],  axis=1, inplace=True)
#combine_df = [train_df, test_df]
sns.countplot(train_df['CabinBand'])


# In[ ]:


train_df[['CabinBand', 'Survived']].groupby(['CabinBand'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### create the Deck feature

# In[ ]:


for dataset in combine_df:
    dataset['Deck'] = dataset.Cabin.str.extract('([A-G])\d+', expand=False)
combine_df = [train_df, test_df]
train_df.Deck.unique()


# In[ ]:


sns.countplot(train_df['Deck'])


# In[ ]:


train_df[['Deck', 'Survived']].groupby(['Deck'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


grid = sns.FacetGrid(train_df, row='Deck', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[ ]:


deckMapping = {'G':1, 'A':2, 'C':3, 'B':4, 'E':5, 'D':6, 'F':7}
for dataset in combine_df:
    dataset['Deck'] = dataset['Deck'].map(deckMapping)
    dataset.Deck.fillna(-1, inplace=True)
    dataset['Deck'] = dataset['Deck'].astype(int)


# In[ ]:


Survived = train_df['Survived']
deckCor = Survived.corr(train_df['Deck'])
deckCor


# In[ ]:


deckCor = Survived.corr(train_df['Deck'], "kendall")
deckCor


# In[ ]:


deckCor = Survived.corr(train_df['Deck'], 'spearman')
deckCor


# In[ ]:


train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)
combine_df = [train_df, test_df]
train_df.head(10)


# ## Handle the Age feature

# In[ ]:


age_band = pd.cut(train_df['Age'], 4)
#train_df['AgeBand'] = pd.cut(train_df['Age'], 4)
sns.countplot(age_band)


# ### I disregaurded the method used in the turorial because it lowered the score

# In[ ]:


for dataset in combine_df:
    dataset['Age'].fillna(-1, inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
#import numpy as np
#guess_ages = np.zeros((2,3))
#for dataset in combine_df:
#    for i in range(0,2):
#        for j in range(0,3):
#            non_null_age_collection_by_class_gender = dataset[(dataset['Sex']==i)& \
#                                                              (dataset['Pclass']==j+1)]['Age'].dropna() 
#            age_guess = non_null_age_collection_by_class_gender.median()
#            guess_ages[i,j] = int(age_guess/0.5 +  0.5) * 0.5
#    print(guess_ages)
#    for i in range(0,2):
#        for j in range(0,3):
#            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) \
#                        & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
#    dataset['Age'] = dataset['Age'].astype(int)
            
train_df.head(10)


# In[ ]:


for dataset in combine_df:
    dataset.loc[ dataset['Age'] == -1, 'Age'] = -1
    dataset.loc[ (dataset['Age'] > -1) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 40), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 60, 'Age'] = 4
train_df.head(10)


# ## Creating Title feature

# In[ ]:


for dataset in combine_df:
    dataset['Title'] = dataset['Name'].str.extract(r'(\w+)\.')
train_df['Title'].value_counts()


# In[ ]:


for dataset in combine_df:    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Rev', 'Dr', 'Countess', 'Jonkheer', 'Don', 'Capt', 'Sir', 'Lady', 'Major', 'Col'], 'Rare')
train_df['Title'].value_counts()


# In[ ]:


train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


title_mapping = {'Mr':1, 'Rare':2, 'Master':3, 'Miss':4, 'Mrs':5}
for dataset in combine_df:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'].fillna(-1, inplace=True)
combine_df = [train_df, test_df]
Survived = train_df['Survived']
titleCor = Survived.corr(train_df['Title'])
titleCor


# In[ ]:


titleCor = Survived.corr(train_df['Title'], method='kendall')
titleCor


# In[ ]:


titleCor = Survived.corr(train_df['Title'], method='spearman')
titleCor


# In[ ]:


train_df = train_df.drop(['Name', 'Ticket'], axis=1)
test_df = test_df.drop(['Name', 'Ticket'], axis=1)
train_df.head()


# ## Setting embarked to int

# In[ ]:


combine_df = [train_df, test_df]
for dataset in combine_df:
    dataset['Embarked'] = dataset['Embarked'].map({'S':1, 'C':2, 'Q':3})
    dataset.Embarked.fillna(0, inplace=True)
    dataset['Embarked'] = dataset['Embarked'].astype(int)
    dataset.fillna(-1, inplace=True)
    
train_df.head()


# ## Changing Fair to FairBand

# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 3)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


for dataset in combine_df:
    dataset.loc[dataset['Fare'] <= 34.885, 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 34.885) & (dataset['Fare'] <= 79.21), 'Fare']  = 2
    dataset.loc[ dataset['Fare'] > 79.2, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
train_df = train_df.drop(['FareBand'], axis=1)
combine_df = [train_df, test_df]
train_df.head(10)


# ## Parch and SibSp to FailySize and IsAlone

# In[ ]:


for dataset in combine_df:
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


for dataset in combine_df:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset.drop(['Parch', 'SibSp'], axis=1,  inplace=True)

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[ ]:


test_df.FamilySize.unique()


# In[ ]:


adult_passengers = train_df.loc[train_df['Age'] >= 2]
grid = sns.FacetGrid(train_df, row='Pclass', size=2.2, aspect=1.6)
grid = grid.map(sns.pointplot, 'FamilySize', 'Survived', 'Sex', palette='deep').set(xticks=[0, 1, 2, 3, 4, 5, 6], xticklabels=[1, 2, 3, 4, 5, 6])
grid.add_legend()


# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# In[ ]:


train_df.head(10)


# ## Getting data ready for modeling

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1)
X_train.shape, Y_train.shape, X_test.shape


# ## Test different models

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns =  ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


from sklearn.svm import SVC, LinearSVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


from sklearn.linear_model import Perceptron
perceptron =  Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
decision_tree= DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# ## Understand the decision tree

# In[ ]:


import graphviz
dot_data = tree.export_graphviz(decision_tree, out_file=None, 
                         feature_names=X_train.columns.values,  
                         class_names=['Survived', 'NotSurvived'],  
                         filled=True, rounded=True,  
                         special_characters=True) 
graph = graphviz.Source(dot_data) 
graph


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# # Random Forest and Decision tree win!

# ## Submit

# In[ ]:


my_submission = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': Y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

